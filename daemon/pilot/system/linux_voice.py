"""The Linux system voice used when neither local voice engine (Kokoro, Pocket TTS) is available.

espeak-ng is the maintained successor of espeak and installs an ``espeak`` alias on most distributions; either is
used. When neither is installed, speech reports one "no voice available" message, logged once per run, and the
conversation carries on in text instead of raising.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable

logger = logging.getLogger("pilot.system.linux_voice")

PROGRAMS = ("espeak-ng", "espeak")
NO_VOICE = (
    "TTS unavailable: no voice available on this system. Install espeak-ng (for example 'sudo apt install "
    "espeak-ng') or a local voice in Settings; Heliox answers in text until then."
)

_warned = False


def espeak_program(which: Callable[[str], str | None] = shutil.which) -> str | None:
    for name in PROGRAMS:
        if which(name):
            return name
    return None


def no_voice_message() -> str:
    global _warned
    if not _warned:
        _warned = True
        logger.warning(NO_VOICE)
    return NO_VOICE
