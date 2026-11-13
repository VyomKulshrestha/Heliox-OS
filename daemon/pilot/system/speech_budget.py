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


# An idle local voice is released only when the computer needs the memory back: less than this free (the voice
# itself holds 1.6-1.9 GB, so other programs are then short), or a computer under SMALL_MEMORY_BYTES in all. Otherwise
# it stays loaded: reloading it takes 20 to 25 s on a laptop CPU, and a person who talks to Heliox every ten minutes
# would meet that every time.
RELEASE_BELOW_FREE_BYTES = 1024**3
RELEASE_RULES = ("memory", "time")  # voice.tts_release: by memory (the default), or by the clock alone


def keep_voice_loaded(free_bytes: int | None, total_bytes: int | None) -> bool:
    """Whether an idle local voice stays loaded on this computer now (voice.tts_release = "memory"): yes unless free
    memory is under RELEASE_BELOW_FREE_BYTES or the computer has under SMALL_MEMORY_BYTES in all. Unknown counts as
    enough."""
    if total_bytes is not None and total_bytes < SMALL_MEMORY_BYTES:
        return False
    return free_bytes is None or free_bytes >= RELEASE_BELOW_FREE_BYTES


def free_memory() -> int | None:
    """Memory available to programs now, in bytes (psutil, the same on Windows, macOS and Linux); None when unknown."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def unavailable_sentence(reason: str) -> str:
    """What the person is told when the local voice could not be used, from the manager's reason."""
    if reason.startswith("This computer has"):
        return reason
    if "not enough free memory" in reason:
        return MEMORY_SHORT
    return VOICE_FAILED
