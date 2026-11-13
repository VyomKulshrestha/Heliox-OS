"""The bench's own root: every Heliox folder inside it, the person's voice settings copied by name, and the speech
models read where they already are, read-only.

A local run outside pytest calls ``enter(root)`` before anything imports pilot.config. Every Heliox folder (settings,
data, state, runtime, ``~/.cache/heliox`` with the wake calibration and the persona, setup's folder) and the
temporary files then live under ``root``, which is kept between runs. ``check_inside(root)`` is asked again before a
listener or a server is built and refuses when any folder resolves outside it.

The person's settings come across only through ``pilot.real_folders.apply_owner_settings``: the ``[model]`` section and
the ``[voice]`` fields in ``OWNER_VOICE``. Only "section.name" is ever printed, never a value.

The speech models are not copied and not fetched. ``read_models_in_place`` points the run at the folders they are
already in: Parakeet's folder as the person's settings name it (or the default one in their data folder), and the
person's data folder for the other downloaded models (the streaming recogniser, Smart Turn, Whisper), all opened
read-only by their loaders. Whisper and the local voice come from the Hugging Face cache, with the hub offline. A
model the run needs that is not on this computer stops the run with its name. Nothing is ever written to those
folders: downloads still go to the bench root's own models folder, and the bench never starts one.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

OWNER_VOICE = tuple(
    f"voice.{name}"
    for name in (
        "language",
        "transcription_engine",
        "whisper_model",
        "final_pass",
        "turn_detection",
        "vad_energy_threshold",
        "vad_adaptive_threshold_enabled",
        "vad_silence_ms",
        "vad_max_utterance_seconds",
        "turn_hold_ms",
        "barge_in_enabled",
        "continuous_conversation_enabled",
        "follow_up_window_seconds",
        "quiet_in_calls",
        "tts_engine",
        "tts_voice",
    )
)
PARAKEET_ID = "parakeet-tdt-0.6b-v2-int8"


ROOT_VARIABLE = "HELIOX_BENCH_ROOT"


def chosen_root(named: str | None, results: Path) -> Path:
    """The bench root: ``--root``, else HELIOX_BENCH_ROOT, else a fresh folder in the results (emptied stores and
    no models: a run then needs the models in place or fetched)."""
    value = (named or os.environ.get(ROOT_VARIABLE, "")).strip()
    return Path(value) if value else results / "data"


class MissingModels(RuntimeError):
    """A speech model the run needs is not on this computer; the bench never downloads one."""


def enter(root: Path) -> Path:
    """Every Heliox folder and the temporary files under ``root`` (kept between runs), checked. Call it before
    anything imports pilot.config; under the test suite's root already, that root is used and checked instead."""
    from pilot.real_folders import use_scratch_folders

    active = use_scratch_folders(Path(root))
    temp = active / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(temp)  # the recorder's utterance files and the voice worker's audio
    tempfile.tempdir = str(temp)
    # No compiled files written next to the code the run imports (the checkout, the environment's packages).
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    check_inside(active)
    return active


def check_inside(root: Path) -> dict[str, Path]:
    """Every Heliox folder pilot.config resolved, each under ``root``; RealFoldersRefused otherwise."""
    from pilot.real_folders import RealFoldersRefused, heliox_folders, where_folders_are

    kind, _ = where_folders_are()
    folders = heliox_folders()
    outside = sorted(name for name, path in folders.items() if not _under(path, root))
    if kind != "test root" or outside:
        raise RealFoldersRefused(
            "the Conversation Bench would use folders outside its own root "
            f"({', '.join(outside) or 'not a scratch root'}); it builds no listener and no server"
        )
    return folders


def _under(path: Path, root: Path) -> bool:
    inner = os.path.normcase(os.path.realpath(path))
    outer = os.path.normcase(os.path.realpath(root))
    return inner == outer or inner.startswith(outer.rstrip(os.sep) + os.sep)


def copy_owner_settings(config: Any) -> list[str]:
    """The person's [model] settings and the [voice] fields in OWNER_VOICE onto ``config``; "section.name" of each."""
    from pilot.real_folders import apply_owner_settings

    return apply_owner_settings(config, ("model", "voice"), only=OWNER_VOICE)


def owner_data_folder() -> Path:
    """The person's real data folder, where Heliox keeps the speech models it downloaded (only ever read here)."""
    from pilot.real_folders import real_budget_db

    return real_budget_db().parent


def _hugging_face_files() -> dict[str, tuple[str, list[str]]]:
    """The Hugging Face cache files each turn-taking model's loader falls back to (read with the hub offline)."""
    from pilot.system import streaming_asr, turn_detector

    return {
        "streaming-zipformer-en": (streaming_asr.REPO, list(streaming_asr.FILES.values())),
        "smart-turn-v3": (turn_detector.REPO, [turn_detector.MODEL_FILE]),
    }


def read_models_in_place(config: Any, *, owner_folder: Path | None = None) -> dict[str, str]:
    """Point the run at the speech models where they are: {model: "bench root" | "in place" | "Hugging Face cache"}.

    The streaming recogniser and Smart Turn always, Parakeet when it is the final pass, Whisper for the wake check
    (and the final pass otherwise). Raises MissingModels naming each one that is nowhere on this computer."""
    from pilot.system import parakeet_asr, speech_models
    from pilot.system.model_presence import faster_whisper_source, hf_cached

    owner = owner_folder or owner_data_folder()
    original = getattr(speech_models.local_path, "__wrapped__", speech_models.local_path)

    def local_path(model_id: str, data_dir: Path | None = None) -> Path | None:
        if data_dir is not None:
            return original(model_id, data_dir)
        return original(model_id) or original(model_id, owner)

    local_path.__wrapped__ = original  # type: ignore[attr-defined]
    speech_models.local_path = local_path

    found: dict[str, str] = {}
    missing: list[str] = []
    voice = config.voice
    for model_id, (repo, files) in _hugging_face_files().items():
        if original(model_id) is not None:
            found[model_id] = "bench root"
        elif original(model_id, owner) is not None:
            found[model_id] = "in place"
        elif all(hf_cached(repo, name) for name in files):
            found[model_id] = "Hugging Face cache"
        else:
            missing.append(model_id)
    if getattr(voice, "final_pass", "") == "parakeet":
        from pilot.real_folders import owner_settings

        named = str(owner_settings("voice").get("parakeet_model_dir") or "").strip()
        folder = Path(named).expanduser() if named else owner / "models" / parakeet_asr.MODEL_NAME
        own = parakeet_asr.model_dir(getattr(voice, "parakeet_model_dir", ""), Path(_data_dir()))
        if not parakeet_asr.missing_files(own):
            found[PARAKEET_ID] = "bench root"
        elif not parakeet_asr.missing_files(folder):
            voice.parakeet_model_dir = str(folder)
            found[PARAKEET_ID] = "in place"
        else:
            missing.append(PARAKEET_ID)
    whisper = str(getattr(voice, "whisper_model", "") or "")
    if whisper:
        source = faster_whisper_source(whisper)  # the patched lookup above: the bench root, then in place
        if source is None:
            missing.append(f"whisper {whisper}")
        else:
            found[f"whisper {whisper}"] = (
                "in place"
                if _under(Path(source), owner)
                else ("bench root" if os.path.isdir(source) else "Hugging Face cache")
            )
    if getattr(voice, "tts_engine", "") == "kokoro_tts":
        from pilot.system import kokoro_tts

        tts_voice = str(getattr(voice, "tts_voice", "") or "")
        wanted = [*kokoro_tts._WEIGHT_FILES, *([f"voices/{tts_voice}.pt"] if tts_voice else [])]
        if all(hf_cached(kokoro_tts.REPO_ID, name) for name in wanted):
            found["kokoro"] = "Hugging Face cache"
        else:
            missing.append("kokoro (the local voice)")
    if missing:
        raise MissingModels("not on this computer, and the bench never downloads: " + ", ".join(sorted(set(missing))))
    return found


def _data_dir() -> Path:
    from pilot.config import DATA_DIR

    return Path(DATA_DIR)
