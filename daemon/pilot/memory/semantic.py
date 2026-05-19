"""Semantic memory utilities for preference extraction and context enrichment."""

from __future__ import annotations

import logging
import re
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.memory.semantic")

PREFERENCE_PATTERNS = [
    (r"(?:set|change|switch to|use|enable)\s+dark\s+(?:mode|theme)", "theme", "dark"),
    (r"(?:set|change|switch to|use|enable)\s+light\s+(?:mode|theme)", "theme", "light"),
    (r"always\s+(?:use|prefer)\s+(\w+)", "preferred_tool", None),
    (r"default\s+(?:editor|terminal)\s+(?:is|to)\s+(\w+)", "default_editor", None),
]


def extract_preferences(user_input: str) -> dict[str, str]:
    """Extract implicit user preferences from natural language input."""
    prefs: dict[str, str] = {}
    lower = user_input.lower()

    for pattern, key, fixed_value in PREFERENCE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            value = fixed_value or match.group(1)
            prefs[key] = value

    return prefs


def summarize_action_history(entries: list[dict[str, Any]], max_entries: int = 10) -> str:
    """Create a concise summary of recent action history for context injection."""
    if not entries:
        return ""

    lines = ["Recent actions:"]
    for entry in entries[:max_entries]:
        status = "OK" if entry.get("success") else "FAILED"
        lines.append(f"  [{status}] {entry.get('user_input', 'N/A')}")

    return "\n".join(lines)
