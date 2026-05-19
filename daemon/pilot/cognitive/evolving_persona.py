"""Evolving Persona Architecture — living neural avatar that adapts daily.

This module implements the fifth revolutionary feature:
- Dynamic persona that changes based on cognitive patterns
- Communication style adapts: concise when stressed, detailed when relaxed
- Daily "neural avatar" updates based on recent patterns
- Learned preferences for UI, tone, and interaction style
- Long-term personality evolution tracking

Architecture:
  Daily Pattern Analysis → Neural Avatar Builder → Communication Adapter → Interaction Optimizer
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.evolving_persona")

# ── Configuration ──

DATA_DIR = Path.home() / ".cache" / "heliox" / "persona"
AVATAR_UPDATE_INTERVAL_HOURS = 24  # Update avatar once per day
PREFERENCE_DECAY = 0.95  # Older preferences fade
MIN_INTERACTIONS_FOR_PATTERN = 10

# Communication styles
STYLE_CONCISE = "concise"
STYLE_DETAILED = "detailed"
STYLE_CASUAL = "casual"
STYLE_FORMAL = "formal"
STYLE_DEFAULT = "balanced"

# Energy levels
ENERGY_HIGH = "high"
ENERGY_MEDIUM = "medium"
ENERGY_LOW = "low"


# ── Data Structures ──


@dataclass
class CommunicationStyle:
    """Communication style preferences."""

    verbosity: float = 0.5  # 0 = very concise, 1 = very detailed
    formality: float = 0.5  # 0 = casual, 1 = formal
    empathy: float = 0.5  # 0 = direct, 1 = empathetic
    speed: float = 0.5  # 0 = slow/explanatory, 1 = fast/efficient

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbosity": round(self.verbosity, 2),
            "formality": round(self.formality, 2),
            "empathy": round(self.empathy, 2),
            "speed": round(self.speed, 2),
        }


@dataclass
class UIPreference:
    """UI interaction preferences."""

    visual_complexity: float = 0.5  # 0 = minimal, 1 = rich
    animation_preference: float = 0.5  # 0 = static, 1 = animated
    notification_level: str = "normal"  # minimal, normal, rich
    color_scheme: str = "auto"  # auto, light, dark
    information_density: float = 0.5  # 0 = sparse, 1 = dense

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_complexity": round(self.visual_complexity, 2),
            "animation_preference": round(self.animation_preference, 2),
            "notification_level": self.notification_level,
            "color_scheme": self.color_scheme,
            "information_density": round(self.information_density, 2),
        }


@dataclass
class DailyPattern:
    """Pattern data for a single day."""

    date: str  # YYYY-MM-DD
    avg_attention: float = 0.5
    avg_stress: float = 0.3
    avg_load: float = 0.4
    energy_level: str = ENERGY_MEDIUM
    dominant_mood: str = "neutral"
    interaction_count: int = 0
    proactive_acceptance_rate: float = 0.5
    communication_style: CommunicationStyle = field(default_factory=CommunicationStyle)
    ui_preference: UIPreference = field(default_factory=UIPreference)


@dataclass
class NeuralAvatar:
    """The living neural avatar that evolves daily."""

    user_id: str
    base_name: str = "JARVIS"
    version: str = "1.0"

    # Current state
    current_style: CommunicationStyle = field(default_factory=CommunicationStyle)
    current_ui: UIPreference = field(default_factory=UIPreference)
    current_energy: str = ENERGY_MEDIUM

    # Historical data
    daily_patterns: list[DailyPattern] = field(default_factory=list)
    weekly_styles: list[CommunicationStyle] = field(default_factory=list)
    weekly_uis: list[UIPreference] = field(default_factory=list)

    # Adaptation
    adaptation_rate: float = 0.3  # How fast to adapt
    stability: float = 0.7  # How stable the persona is

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    total_interactions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "base_name": self.base_name,
            "version": self.version,
            "current_style": self.current_style.to_dict(),
            "current_ui": self.current_ui.to_dict(),
            "current_energy": self.current_energy,
            "adaptation_rate": round(self.adaptation_rate, 2),
            "stability": round(self.stability, 2),
            "total_interactions": self.total_interactions,
            "last_update": datetime.fromtimestamp(self.last_update).isoformat(),
        }


# ── Storage ──


class PersonaStore:
    """Persists persona data to disk."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _get_avatar_path(self, user_id: str) -> Path:
        return self._data_dir / f"avatar_{user_id}.json"

    def load_avatar(self, user_id: str) -> NeuralAvatar | None:
        path = self._get_avatar_path(user_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Reconstruct nested objects
            if "current_style" in data:
                data["current_style"] = CommunicationStyle(**data["current_style"])
            if "current_ui" in data:
                data["current_ui"] = UIPreference(**data["current_ui"])
            if "daily_patterns" in data:
                patterns = []
                for p in data["daily_patterns"]:
                    pat = DailyPattern(**p)
                    if "communication_style" in p:
                        pat.communication_style = CommunicationStyle(**p["communication_style"])
                    if "ui_preference" in p:
                        pat.ui_preference = UIPreference(**p["ui_preference"])
                    patterns.append(pat)
                data["daily_patterns"] = patterns
            return NeuralAvatar(**data)
        except Exception as e:
            logger.warning("Failed to load avatar: %s", e)
            return None

    def save_avatar(self, avatar: NeuralAvatar) -> None:
        path = self._get_avatar_path(avatar.user_id)
        data = {
            "user_id": avatar.user_id,
            "base_name": avatar.base_name,
            "version": avatar.version,
            "current_style": avatar.current_style.to_dict(),
            "current_ui": avatar.current_ui.to_dict(),
            "current_energy": avatar.current_energy,
            "daily_patterns": [
                {
                    "date": p.date,
                    "avg_attention": p.avg_attention,
                    "avg_stress": p.avg_stress,
                    "avg_load": p.avg_load,
                    "energy_level": p.energy_level,
                    "dominant_mood": p.dominant_mood,
                    "interaction_count": p.interaction_count,
                    "proactive_acceptance_rate": p.proactive_acceptance_rate,
                    "communication_style": p.communication_style.to_dict(),
                    "ui_preference": p.ui_preference.to_dict(),
                }
                for p in avatar.daily_patterns
            ],
            "adaptation_rate": avatar.adaptation_rate,
            "stability": avatar.stability,
            "created_at": avatar.created_at,
            "last_update": avatar.last_update,
            "total_interactions": avatar.total_interactions,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Core Engine ──


class EvolvingPersonaEngine:
    """Engine that evolves the persona based on cognitive patterns."""

    def __init__(self, user_id: str = "default", store: PersonaStore | None = None):
        self._user_id = user_id
        self._store = store or PersonaStore()
        self._avatar = self._load_or_create_avatar()

        # Real-time adaptation buffer
        self._today_interactions: list[dict[str, Any]] = []
        self._current_daily_pattern: DailyPattern | None = None

    def _load_or_create_avatar(self) -> NeuralAvatar:
        avatar = self._store.load_avatar(self._user_id)
        if avatar:
            return avatar
        return NeuralAvatar(user_id=self._user_id)

    # ── Daily Pattern Recording ──

    def record_interaction(
        self,
        attention: float,
        stress: float,
        load: float,
        user_response: str | None = None,  # "accepted", "dismissed", "modified"
    ) -> None:
        """Record an interaction to build today's pattern."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Initialize today's pattern if needed
        if not self._current_daily_pattern or self._current_daily_pattern.date != date_str:
            self._current_daily_pattern = DailyPattern(date=date_str)

        # Update running averages
        p = self._current_daily_pattern
        n = p.interaction_count + 1

        p.avg_attention = (p.avg_attention * p.interaction_count + attention) / n
        p.avg_stress = (p.avg_stress * p.interaction_count + stress) / n
        p.avg_load = (p.avg_load * p.interaction_count + load) / n
        p.interaction_count = n

        # Track proactive acceptance
        if user_response:
            old_rate = p.proactive_acceptance_rate
            if user_response == "accepted":
                p.proactive_acceptance_rate = (old_rate * (n - 1) + 1.0) / n
            elif user_response == "dismissed":
                p.proactive_acceptance_rate = (old_rate * (n - 1) + 0.0) / n

        # Determine energy level
        if p.avg_attention > 0.7 and p.avg_stress < 0.4:
            p.energy_level = ENERGY_HIGH
        elif p.avg_attention < 0.4 or p.avg_stress > 0.6:
            p.energy_level = ENERGY_LOW
        else:
            p.energy_level = ENERGY_MEDIUM

        # Determine mood
        if p.avg_stress > 0.6:
            p.dominant_mood = "stressed"
        elif p.avg_stress < 0.3 and p.avg_attention > 0.6:
            p.dominant_mood = "focused"
        elif p.avg_attention < 0.4:
            p.dominant_mood = "tired"
        else:
            p.dominant_mood = "neutral"

        self._avatar.total_interactions += 1

        # Store interaction
        self._today_interactions.append(
            {
                "timestamp": time.time(),
                "attention": attention,
                "stress": stress,
                "load": load,
                "user_response": user_response,
            }
        )

    def _update_communication_style(self) -> CommunicationStyle:
        """Update communication style based on current state."""
        if not self._current_daily_pattern:
            return self._avatar.current_style

        stress = self._current_daily_pattern.avg_stress
        load = self._current_daily_pattern.avg_load
        energy = self._current_daily_pattern.energy_level

        # High stress or load → more concise, more empathetic
        verbosity = 0.5
        if stress > 0.6 or load > 0.7:
            verbosity = 0.2  # Concise
        elif stress < 0.3 and load < 0.5:
            verbosity = 0.7  # Detailed

        # Energy affects speed and formality
        speed = 0.5
        formality = 0.5
        if energy == ENERGY_HIGH:
            speed = 0.7
            formality = 0.6
        elif energy == ENERGY_LOW:
            speed = 0.3
            formality = 0.3

        # Empathy: higher when stressed
        empathy = 0.5 + (stress * 0.3)

        # Apply with adaptation rate (blend with current)
        alpha = self._avatar.adaptation_rate

        style = CommunicationStyle(
            verbosity=alpha * verbosity + (1 - alpha) * self._avatar.current_style.verbosity,
            formality=alpha * formality + (1 - alpha) * self._avatar.current_style.formality,
            empathy=alpha * empathy + (1 - alpha) * self._avatar.current_style.empathy,
            speed=alpha * speed + (1 - alpha) * self._avatar.current_style.speed,
        )

        return style

    def _update_ui_preference(self) -> UIPreference:
        """Update UI preferences based on current state."""
        if not self._current_daily_pattern:
            return self._avatar.current_ui

        stress = self._current_daily_pattern.avg_stress
        load = self._current_daily_pattern.avg_load

        # High stress → simpler UI
        visual_complexity = 0.5
        information_density = 0.5
        notification_level = "normal"

        if stress > 0.6 or load > 0.7:
            visual_complexity = 0.2
            information_density = 0.3
            notification_level = "minimal"
        elif stress < 0.3 and load < 0.5:
            visual_complexity = 0.7
            information_density = 0.6
            notification_level = "rich"

        # Apply with adaptation rate
        alpha = self._avatar.adaptation_rate

        ui = UIPreference(
            visual_complexity=alpha * visual_complexity + (1 - alpha) * self._avatar.current_ui.visual_complexity,
            information_density=alpha * information_density + (1 - alpha) * self._avatar.current_ui.information_density,
            notification_level=notification_level,
            animation_preference=self._avatar.current_ui.animation_preference,
            color_scheme=self._avatar.current_ui.color_scheme,
        )

        return ui

    # ── Avatar Update ──

    def update_avatar(self) -> None:
        """Update the neural avatar with current state."""
        # Update communication style
        self._avatar.current_style = self._update_communication_style()

        # Update UI preference
        self._avatar.current_ui = self._update_ui_preference()

        # Update energy
        if self._current_daily_pattern:
            self._avatar.current_energy = self._current_daily_pattern.energy_level

        # Store today's pattern
        if self._current_daily_pattern:
            self._avatar.daily_patterns.append(self._current_daily_pattern)

        # Keep only last 30 days
        if len(self._avatar.daily_patterns) > 30:
            self._avatar.daily_patterns = self._avatar.daily_patterns[-30:]

        # Update weekly averages
        self._avatar.weekly_styles.append(self._avatar.current_style)
        self._avatar.weekly_uis.append(self._avatar.current_ui)

        if len(self._avatar.weekly_styles) > 7:
            self._avatar.weekly_styles = self._avatar.weekly_styles[-7:]
            self._avatar.weekly_uis = self._avatar.weekly_uis[-7:]

        self._avatar.last_update = time.time()

        # Persist
        self._store.save_avatar(self._avatar)

        logger.info(
            "Neural avatar updated: style=%s, energy=%s, interactions=%d",
            self._avatar.current_style.to_dict(),
            self._avatar.current_energy,
            self._avatar.total_interactions,
        )

    # ── Communication Generation ──

    def get_greeting(self) -> str:
        """Generate an appropriate greeting based on current state."""
        hour = datetime.now().hour
        energy = self._avatar.current_energy
        stress = self._current_daily_pattern.avg_stress if self._current_daily_pattern else 0.5

        # Time-based prefix
        if 6 <= hour < 12:
            time_greet = "Good morning"
        elif 12 <= hour < 17:
            time_greet = "Good afternoon"
        elif 17 <= hour < 21:
            time_greet = "Good evening"
        else:
            time_greet = "Hello"

        # Energy-based modifier
        if energy == ENERGY_HIGH:
            energy_mod = "! Ready to tackle anything?"
        elif energy == ENERGY_LOW:
            energy_mod = ". Taking it easy today?"
        else:
            energy_mod = ". How can I help?"

        # Stress-aware modifier
        if stress > 0.6:
            energy_mod = ". I'll keep things simple - just say the word."

        return f"{time_greet}{energy_mod}"

    def format_response(
        self,
        base_message: str,
        include_details: bool = True,
    ) -> str:
        """Format a response based on current communication style."""
        style = self._avatar.current_style

        if style.verbosity < 0.3:
            # Very concise
            return base_message[:100] + ("..." if len(base_message) > 100 else "")

        if style.verbosity > 0.7 and include_details:
            # Detailed - add context
            return base_message + "\n\n[Additional context available if needed]"

        return base_message

    def get_ui_config(self) -> dict[str, Any]:
        """Get UI configuration based on current preferences."""
        ui = self._avatar.current_ui

        return {
            "visual_complexity": ui.visual_complexity,
            "information_density": ui.information_density,
            "notification_level": ui.notification_level,
            "animations": ui.animation_preference > 0.5,
            "color_scheme": ui.color_scheme,
            "empathy_mode": self._avatar.current_style.empathy > 0.6,
        }

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "avatar": self._avatar.to_dict(),
            "today_interactions": len(self._today_interactions),
            "current_pattern": {
                "date": self._current_daily_pattern.date if self._current_daily_pattern else None,
                "energy": self._current_daily_pattern.energy_level if self._current_daily_pattern else None,
                "interaction_count": self._current_daily_pattern.interaction_count
                if self._current_daily_pattern
                else 0,
            }
            if self._current_daily_pattern
            else None,
        }
