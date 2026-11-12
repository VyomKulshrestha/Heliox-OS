"""Which speech models are on this machine, without ever downloading one.

A model is used when it is in Heliox's data folder (downloaded through the app, pilot.system.speech_models) or was
already in the machine's caches before (Hugging Face's for Faster-Whisper, the streaming model and Smart Turn;
OpenAI Whisper's own ``~/.cache/whisper``), so an existing install keeps working without fetching again. Nothing
here, and nothing that asks here first, reaches the network.

Without a recogniser Heliox still starts listening; the voice bar says that speech recognition needs a model and
points to Settings, Voice. Without the streaming model there are no live captions; without Smart Turn the end of a
sentence is judged by silence alone.
"""

from __future__ import annotations

import os
from pathlib import Path


class SpeechModelMissing(RuntimeError):
    """A model that is neither in Heliox's data folder nor already in a local cache. Never fetched by itself."""


FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small.en": "Systran/faster-whisper-small.en",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
}
DOWNLOADED = {"base": "whisper-base"}  # whisper_model -> speech_models id offered in the app


def hf_cached(repo: str, filename: str) -> str | None:
    """The path of a file already in the Hugging Face cache, or None. Reads the cache only."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    try:
        found = try_to_load_from_cache(repo, filename)
    except Exception:
        return None
    return found if isinstance(found, str) and os.path.isfile(found) else None


def openai_whisper_cached(model_name: str) -> str | None:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(root, "whisper", f"{model_name}.pt")
    return path if os.path.isfile(path) else None


def faster_whisper_source(model_name: str) -> str | None:
    """Where Faster-Whisper ``model_name`` can be loaded from without a download: a folder path, the model name
    (meaning "from the Hugging Face cache, local files only"), or None when it is not on this machine."""
    from pilot.system.speech_models import local_path

    model_id = DOWNLOADED.get(model_name)
    folder = local_path(model_id) if model_id else None
    if folder is not None:
        return str(folder)
    if os.path.isdir(model_name):
        return model_name  # a folder chosen by hand
    repo = FASTER_WHISPER_REPOS.get(model_name)
    if repo and hf_cached(repo, "model.bin"):
        return model_name
    return None


def whisper_present(model_name: str) -> bool:
    return faster_whisper_source(model_name) is not None or openai_whisper_cached(model_name) is not None


def parakeet_present(folder: Path) -> bool:
    from pilot.system.parakeet_asr import missing_files

    return not missing_files(folder)


def recognisers(whisper_model: str, parakeet_folder: Path | None, final_pass: str) -> dict[str, bool]:
    """Which recognisers can run now: {"whisper": ..., "parakeet": ...} (Parakeet only when chosen)."""
    return {
        "whisper": whisper_present(whisper_model),
        "parakeet": final_pass == "parakeet" and parakeet_folder is not None and parakeet_present(parakeet_folder),
    }


MISSING_MESSAGE = (
    "Speech recognition needs a model, which Heliox has not downloaded. Open Settings, Voice to download one "
    "(Whisper base, 148 MB, is smaller and faster; choose Parakeet for the best English). Typing works meanwhile."
)


def wake_route(present: dict[str, bool]) -> str:
    """How the wake phrase is found with the recognisers present, in a sentence for Settings and the log."""
    if present.get("whisper") and present.get("parakeet"):
        return "Parakeet hears the words and Whisper listens for the wake phrase."
    if present.get("whisper"):
        return "Whisper hears the words and the wake phrase."
    if present.get("parakeet"):
        return (
            "Parakeet hears the words and the wake phrase; it often writes the name Heliox differently, so the "
            "wake phrase is missed more often. Download Whisper base as well for a reliable wake phrase."
        )
    return "No recogniser: Heliox cannot hear the wake phrase until a speech model is downloaded."
