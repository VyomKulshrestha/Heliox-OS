"""The person's conversation settings for a paid run, read and copied (pilot.real_folders copies [model] only).

The fast model a paid run talks to is chosen by the [conversation] section as well as [model]: the model for each
provider (``claude_model``, ...) and the reply length. This reads that section from the person's settings, the way
pilot.real_folders.owner_model_settings reads [model], and copies what this version knows onto a config of the run's own
folder. Nothing here builds a config over the person's file, so nothing could save it.
"""

from __future__ import annotations

import tomllib
from typing import Any

from pilot import real_folders


def owner_conversation_settings() -> dict[str, Any]:
    """The [conversation] section of the person's settings; empty when the file is missing or cannot be read."""
    path = real_folders._owner_folder("XDG_CONFIG_HOME", ".config") / "config.toml"
    try:
        section = tomllib.loads(path.read_text(encoding="utf-8")).get("conversation", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    return dict(section) if isinstance(section, dict) else {}


def apply_owner_settings(config: Any) -> list[str]:
    """The person's [model] and [conversation] settings copied onto ``config`` (one whose files are in the run's own
    folder). Returns the names copied."""
    copied = [f"model.{name}" for name in real_folders.apply_owner_model_settings(config)]
    for name, value in owner_conversation_settings().items():
        if hasattr(config.conversation, name):
            setattr(config.conversation, name, value)
            copied.append(f"conversation.{name}")
    return copied
