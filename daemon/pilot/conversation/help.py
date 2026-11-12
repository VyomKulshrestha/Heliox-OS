"""'What can I say?' answered in two spoken sentences, with three examples this person can use right now.

The examples are chosen from what works on this machine at this moment: the kind of model set up (a request to act
works with none, a question needs one), the operating system (which app is safe to name), and whether Heliox Live
is on. "More" within two minutes gives the next three. The command palette, opened at the same time, keeps the
full list for anyone who wants to read it.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

MORE_WITHIN_SECONDS = 120.0
MORE_REQUEST = re.compile(r"^(?:more|tell me more|what else|and what else|anything else|more please|say more)$")


def _app_example(platform: str) -> str:
    if platform == "win32":
        return "open Notepad"
    if platform == "darwin":
        return "open Notes"
    return "open the terminal"


def examples(*, setup: str, live: bool, platform: str | None = None) -> list[str]:
    """Spoken examples, best first. ``setup``: "claude", "cloud", "local", "subscription" or "none"."""
    found = [_app_example(platform or sys.platform), "remind me in 20 minutes to stretch"]
    if setup != "none":
        found.append("why might my build be slow?")
    found += ["what's on my list?", "mute", "search the web for train times"]
    if live:
        found.append("why did you say that?")  # after Live points something out
    found += ["start a pomodoro", "what did you do today?", "undo that"]
    return found


def _sentence(chosen: list[str]) -> str:
    quoted = [f'"{example}"' for example in chosen]
    if len(quoted) == 1:
        return f"You can also say {quoted[0]}."
    return f"You can say things like {', '.join(quoted[:-1])}, or {quoted[-1]}."


class SpokenHelp:
    """The pages of examples for one server, and the "more" that follows."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._asked_at = float("-inf")
        self._page = 0

    def first(self, *, setup: str, live: bool, platform: str | None = None) -> str:
        self._asked_at, self._page = self._clock(), 1
        chosen = examples(setup=setup, live=live, platform=platform)[:3]
        return f'{_sentence(chosen)} Say "more" for others.'

    def more(self, text: str, *, setup: str, live: bool, platform: str | None = None) -> str | None:
        """The next page when the person says "more" soon after asking; None when this is not that."""
        normalized = " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).split())
        if not MORE_REQUEST.match(normalized) or self._clock() - self._asked_at > MORE_WITHIN_SECONDS:
            return None
        every = examples(setup=setup, live=live, platform=platform)
        chosen = every[self._page * 3 : self._page * 3 + 3]
        self._page += 1
        self._asked_at = self._clock()
        if not chosen:
            return "That's most of it. The full list is in the command palette, Control K."
        return _sentence(chosen)


def help_for(server: Any) -> SpokenHelp:
    existing = getattr(server, "_spoken_help", None)
    if existing is None:
        existing = server._spoken_help = SpokenHelp()
    return existing


def server_state(server: Any) -> dict[str, Any]:
    from pilot.conversation.fast_model import describe_setup

    config = getattr(server, "config", None)
    return {
        "setup": describe_setup(config) if config is not None else "none",
        "live": bool(getattr(getattr(config, "live", None), "enabled", False)),
    }
