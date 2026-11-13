"""Whether this computer can afford Heliox's own voice beside everything else the person runs.

Measured on a 16 GB laptop: the local voice (Kokoro, its own process) holds about 1.6-1.9 GB once loaded, and with
both recognisers about 2.5 GB; free memory fell from 2.7 to 0.8 GB while it loaded. On a computer with 8 GB and a
browser open that pushes other programs into swap. So on a computer with less than SMALL_MEMORY_BYTES of memory in
all, Heliox speaks with the system voice by itself and says so once, with the setting that changes it
(``voice.light_on_small_memory``, on by default). Whatever the computer, the voice still needs VOICE_HOST_RAM_BYTES
free when it loads (pilot.system.local_tts_worker), and a load that fails falls back to the system voice.

Pure: the caller passes the total memory, so tests can say 4, 8 or 16 GB.
"""

from __future__ import annotations

SMALL_MEMORY_BYTES = 10 * 1024**3  # an "8 GB" computer reports about 7.6-7.9 GiB; 12 and 16 GB computers keep the voice
SETTING = "Settings, Voice, “Use the system voice on computers with little memory”"
MEMORY_SHORT = (
    "There isn't enough free memory for Heliox's own voice right now, so the system voice speaks. Closing some "
    "programs frees it; Heliox tries again in two minutes."
)
VOICE_FAILED = "Heliox's own voice is not working, so the system voice speaks. Heliox tries again in two minutes."


def total_memory() -> int | None:
    """This computer's memory in bytes (psutil, the same on Windows, macOS and Linux); None when it cannot tell."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def light_voice_reason(total_bytes: int | None, enabled: bool) -> str:
    """The sentence to say once when Heliox should speak with the system voice on this computer, "" when not."""
    if not enabled or total_bytes is None or total_bytes >= SMALL_MEMORY_BYTES:
        return ""
    gigabytes = max(1, round(total_bytes / 1024**3))
    return (
        f"This computer has {gigabytes} GB of memory, so Heliox speaks with the system voice to leave room for your "
        f"other programs. To use Heliox's own voice anyway, turn off {SETTING}."
    )


def unavailable_sentence(reason: str) -> str:
    """What the person is told when the local voice could not be used, from the manager's reason."""
    if reason.startswith("This computer has"):
        return reason
    if "not enough free memory" in reason:
        return MEMORY_SHORT
    return VOICE_FAILED
