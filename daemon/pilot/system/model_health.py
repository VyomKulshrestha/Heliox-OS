"""One list that says, for each local model, whether it is loaded, where it runs and what last went wrong.

The services strip and the bench read it from ``autonomy_review_status`` (field ``local_models``). It is assembled
from what the components already report (Gnomon's reviewer status, the local voice manager, the voice listener's
stats, the OCR readers) and the GPU arbiter's fault log; nothing here loads or probes a model.

Each entry: ``name``, ``state`` ("loaded" | "loading" | "not_loaded" | "unavailable" | "off" | "not_supported"),
``reason`` (plain words when the state is not "loaded"), ``device`` ("cuda", "mps", "cpu", "system" or None),
``process`` ("daemon", "worker" or "system": where a fault would land) and ``last_fault`` ({at, message} or None).
The same keys and words on Windows, macOS and Linux; "not_supported" says why when that is the truth.
"""

from __future__ import annotations

import shutil
import sys
from typing import Any

_GNOMON_STATES = {
    "ready": "loaded",
    "loading": "loading",
    "cold": "not_loaded",
    "unavailable": "unavailable",
    "disabled": "off",
}


def _entry(
    name: str,
    state: str,
    *,
    reason: str = "",
    device: str | None = None,
    process: str = "daemon",
    last_fault: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "state": state,
        "reason": reason,
        "device": device,
        "process": process,
        "last_fault": last_fault,
    }


def gnomon_entry(status: dict[str, Any]) -> dict[str, Any]:
    state = _GNOMON_STATES.get(str(status.get("state")), "unavailable")
    if state == "unavailable" and status.get("supported") is False:
        state = "not_supported"
    reason = str(status.get("reason") or "")
    if state == "off":
        reason = "Switched off in Settings."
    elif state == "not_loaded":
        reason = "Loads in the background when a plan needs a review."
    return _entry(
        "Gnomon",
        state,
        reason=reason,
        device=status.get("device"),
        process="worker" if status.get("process") == "separate" else "daemon",
        last_fault=status.get("last_fault"),
    )


def voice_entry(voice_config: Any, manager: Any, last_fault: dict[str, Any] | None) -> dict[str, Any]:
    engine = str(getattr(voice_config, "tts_engine", ""))
    if engine not in {"kokoro_tts", "pocket_tts"}:
        return _entry("Voice", "loaded", device="system", process="system", reason="The system voice.")
    configured = str(getattr(voice_config, "tts_device", "auto"))
    same = bool(manager.running and manager.engine == engine)
    device = (getattr(manager, "effective_device", None) or configured) if same else None
    load_task = getattr(manager, "_load_task", None)
    if getattr(manager, "failing", False):
        state, reason = "unavailable", "Speaking with the system voice until the local voice recovers."
    elif same and manager.loaded:
        state, reason = "loaded", ""
    elif same and load_task is not None and not load_task.done():
        state, reason = "loading", "Getting the voice ready."
    else:
        state, reason = "not_loaded", "Loads when someone speaks to Heliox."
    if device == "cpu" and configured != "cpu":
        reason = (reason + " On the CPU: the graphics card was hot, busy or switched off when it started.").strip()
    return _entry("Voice", state, reason=reason, device=device, process="worker", last_fault=last_fault)


def recogniser_entries(stats: dict[str, Any] | None, whisper_devices: list[str]) -> list[dict[str, Any]]:
    if not stats or not stats.get("running"):
        reason = "The voice listener is not running."
        return [_entry(name, "off", reason=reason) for name in ("Streaming recogniser", "Turn detector", "Final pass")]
    entries = []
    for name, ready_key, error_key in (
        ("Streaming recogniser", "streaming_ready", "streaming_error"),
        ("Turn detector", "turn_detector_ready", "turn_detector_error"),
    ):
        error = str(stats.get(error_key) or "")
        if stats.get(ready_key):
            entries.append(_entry(name, "loaded", device="cpu"))
        elif error:
            entries.append(_entry(name, "unavailable", reason=error))
        else:
            entries.append(_entry(name, "loading", reason="Loads when the listener starts."))
    final_pass = str(stats.get("final_pass") or "whisper")
    notice = str(stats.get("final_pass_notice") or "")
    device = whisper_devices[0] if whisper_devices else None
    state = "loaded" if whisper_devices else "not_loaded"
    # Why Whisper is on the CPU or the graphics card (the GPU arbiter's reason), unless a notice says more.
    reason = notice or str(stats.get("whisper_device_reason") or "")
    if final_pass == "parakeet":
        entries.append(_entry("Final pass (Parakeet)", "loaded", device="cpu"))
        # Whisper still reads each utterance for the wake phrase alongside Parakeet.
        entries.append(_entry("Wake check (Whisper)", state, reason=reason, device=device))
    else:
        entries.append(_entry("Final pass (Whisper)", state, reason=reason, device=device))
    return entries


def whisper_devices() -> list[str]:
    """Devices of the Whisper models the voice module has loaded in this process (read-only)."""
    voice = sys.modules.get("pilot.system.voice")
    cache = getattr(voice, "_faster_whisper_model_cache", None) if voice is not None else None
    if not isinstance(cache, dict):
        return []
    return sorted({str(key[1]) for key in cache if isinstance(key, tuple) and len(key) > 1})


def ocr_entry(platform_name: str, easyocr_devices: list[str], last_fault: dict[str, Any] | None) -> dict[str, Any]:
    """Screen text reading. The system engines run as separate programs; EasyOCR, when used, runs in the daemon."""
    if easyocr_devices:
        device = "cuda" if "gpu" in easyocr_devices else "cpu"
        return _entry("Screen text (EasyOCR)", "loaded", device=device, last_fault=last_fault)
    if platform_name == "windows":
        return _entry("Screen text", "loaded", device="system", process="system", reason="Windows OCR.")
    if platform_name == "macos":
        return _entry("Screen text", "loaded", device="system", process="system", reason="macOS Vision.")
    if shutil.which("tesseract"):
        return _entry("Screen text", "loaded", device="cpu", process="system", reason="Tesseract.")
    return _entry(
        "Screen text",
        "not_supported",
        reason="No text reader on this system: install Tesseract (or EasyOCR) to read text on screen.",
        last_fault=last_fault,
    )


def easyocr_devices() -> list[str]:
    vision = sys.modules.get("pilot.system.vision")
    reader_holder = getattr(vision, "_ocr_easyocr", None) if vision is not None else None
    if reader_holder is None:
        return []
    names = [name for name in vars(reader_holder) if name.startswith("_easyocr_reader_")]
    return sorted({name.rsplit("_", 1)[-1] for name in names})


def platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def local_models(
    *,
    gnomon_status: dict[str, Any],
    voice_config: Any,
    voice_manager: Any,
    listener_stats: dict[str, Any] | None,
    arbiter: Any,
) -> list[dict[str, Any]]:
    def fault(component: str) -> dict[str, Any] | None:
        last = arbiter.last_fault(component)
        return last.to_dict() if last is not None else None

    return [
        gnomon_entry(gnomon_status),
        voice_entry(voice_config, voice_manager, fault("voice")),
        *recogniser_entries(listener_stats, whisper_devices()),
        ocr_entry(platform_name(), easyocr_devices(), fault("ocr")),
    ]
