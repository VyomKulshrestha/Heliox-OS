"""Changelog & Feature Announcements — notifies users of new features."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.changelog")

VERSION = "0.6.0"

CHANGELOG = {
    "0.6.0": {
        "title": "Revolutionary Cognitive Intelligence Features",
        "date": "2026-04-08",
        "features": [
            {
                "name": "Adaptive Biometric Learning Loop",
                "description": "Tracks your patterns over weeks. Learns when you're most productive.",
                "jarvis_announce": "I've learned your patterns. I know when you work best.",
            },
            {
                "name": "Ambient Intelligence Mode",
                "description": "Proactive suggestions. 'You've been working for 2 hours - take a break?'",
                "jarvis_announce": "I'll proactively help before you get overwhelmed.",
            },
            {
                "name": "Multi-Modal Neural Bridge",
                "description": "Webcam, audio, keyboard dynamics. Builds neural workspace.",
                "jarvis_announce": "I now understand your focus through multiple inputs.",
            },
            {
                "name": "Cognitive Offloading",
                "description": "Memory anchors when load > 80%. Remembers complex workflows.",
                "jarvis_announce": "I'll remember complex tasks so you don't have to.",
            },
            {
                "name": "Evolving Persona",
                "description": "Communication adapts: concise when stressed, detailed when relaxed.",
                "jarvis_announce": "My communication adapts to your cognitive state.",
            },
            {
                "name": "Cross-Device Handoff",
                "description": "Sync state to cloud. Continue on mobile with context.",
                "jarvis_announce": "Your context follows you across devices.",
            },
            {
                "name": "Quantum-Ready Architecture",
                "description": "Model-agnostic. Swap in future models without touching call sites.",
                "jarvis_announce": "I'm built for the future of AI.",
            },
        ],
        "summary": "7 biologically-inspired AI features powered by a lightweight cognitive engine",
    },
    "0.5.1": {
        "title": "Cognitive Engine Integrations",
        "date": "2026-04-01",
        "features": [
            "Neural Cognitive HUD",
            "Dynamic TTS Stress-Pacing",
            "Neuro-Safe Destructive Gate",
            "Subconscious Persona Fingerprint",
            "Attention-Optimized Notifications",
            "ReAct Neural Cost Estimator",
            "JARVIS Intent Classifier",
        ],
        "summary": "Core cognitive engine integrations",
    },
}


def get_state_dir() -> Path:
    from pilot.config import STATE_DIR

    return STATE_DIR


def get_last_version() -> str | None:
    state_dir = get_state_dir()
    version_file = state_dir / "last_version.txt"
    if version_file.exists():
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def set_last_version(version: str) -> None:
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    version_file = state_dir / "last_version.txt"
    version_file.write_text(version, encoding="utf-8")


def check_for_updates() -> list[dict[str, Any]]:
    """Check if there are new features since last run."""
    last_version = get_last_version()
    current_version = VERSION

    if last_version is None:
        return get_full_changelog()

    if last_version == current_version:
        return []

    new_features = []
    for ver in CHANGELOG:
        if _compare_versions(ver, last_version) > 0:
            new_features.extend(CHANGELOG[ver]["features"])

    return new_features


def get_full_changelog() -> list[dict[str, Any]]:
    features = []
    for ver in CHANGELOG:
        ver_features = CHANGELOG[ver]["features"]
        # Handle both dict features and string features
        if ver_features and isinstance(ver_features[0], dict):
            for feat in ver_features:
                feat_copy = dict(feat)  # Copy to avoid mutation
                feat_copy["version"] = ver
                feat_copy["date"] = CHANGELOG[ver]["date"]
                features.append(feat_copy)
        else:
            # String feature - convert to dict
            for feat_name in ver_features:
                features.append(
                    {
                        "name": feat_name,
                        "version": ver,
                        "date": CHANGELOG[ver]["date"],
                    }
                )
    return features


def _compare_versions(v1: str, v2: str) -> int:
    """Compare semantic versions. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
    parts1 = [int(x) for x in v1.split(".")]
    parts2 = [int(x) for x in v2.split(".")]
    for p1, p2 in zip(parts1, parts2):
        if p1 > p2:
            return 1
        elif p1 < p2:
            return -1
    return 0


def get_welcome_message() -> dict[str, Any]:
    """Get the welcome message for new users."""
    return {
        "title": "Welcome to Heliox OS",
        "version": VERSION,
        "tagline": "Your biologically-inspired AI assistant",
        "features": get_full_changelog()[:3],
    }


def announce_new_features() -> str:
    """Generate JARVIS announcement for new features."""
    new_features = check_for_updates()

    if not new_features:
        return ""

    lines = [
        "Welcome to Heliox OS version " + VERSION + ".",
    ]

    for feat in new_features[:3]:
        if "jarvis_announce" in feat:
            lines.append(feat["jarvis_announce"])

    lines.append("Say 'What can you do?' to learn more.")

    return " ".join(lines)


def mark_version_seen() -> None:
    """Mark that user has seen current version."""
    set_last_version(VERSION)


def get_cognitive_status() -> dict[str, Any]:
    """Get brief cognitive feature status for HUD."""
    new_features = check_for_updates()
    return {
        "new_features_available": len(new_features) > 0,
        "new_feature_count": len(new_features),
        "version": VERSION,
        "cognitive_engine_available": True,
    }
