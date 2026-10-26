"""Small helpers: strong passwords, coin flips, dice, random picks and a stopwatch.

"Generate a password" makes one with the operating system's secure random source and
puts it on the clipboard. It is never spoken, shown in chat, written to disk or kept in
clipboard history (which skips anything that looks like a password). "Flip a coin",
"roll a die", "pick a number between 1 and 100", "choose between pizza and sushi" and
"start a stopwatch" are answered in code.
"""

from __future__ import annotations

import re
import secrets
import string
import time
from collections.abc import Awaitable, Callable
from typing import Any

PASSWORD_REQUEST = re.compile(
    r"^(?:please )?(?:generate|make|create|give) (?:me )?(?:a |an )?(?:new )?(?:(?P<length>\d{1,3})(?: |-)?(?:character|char|letter) )?"
    r"(?:strong |secure |random )?(?:password|passcode)$"
)
COIN_REQUEST = re.compile(r"^(?:please )?(?:flip|toss) a coin$|^heads or tails$")
DIE_REQUEST = re.compile(
    r"^(?:please )?roll (?:a |an |one |the )?(?:(?:die|dice)|d(?P<sides>\d{1,3}))$|^roll (?P<count>two|2|three|3) dice$"
)
NUMBER_REQUEST = re.compile(
    r"^(?:please )?(?:pick|choose|give me) a (?:random )?number (?:between|from) (?P<low>-?\d+) (?:and|to) (?P<high>-?\d+)$"
)
CHOICE_REQUEST = re.compile(
    r"^(?:please )?(?:choose|pick|decide) (?:between|for me between|one of|for me|one) (?P<options>.+ or .+)$"
)
STOPWATCH_START = re.compile(r"^(?:please )?(?:start|begin) (?:a |the )?stopwatch$|^start timing(?: me)?$")
STOPWATCH_CHECK = re.compile(
    r"^(?:how long has it been|what s the stopwatch at|check the stopwatch|how long (?:have i been going|has the stopwatch been running))$"
)
STOPWATCH_STOP = re.compile(r"^(?:please )?stop (?:the )?(?:stopwatch|timing)$")
_SYMBOLS = "!@#$%^&*-_=+?"
_COUNTS = {"two": 2, "three": 3}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", text.casefold()).split())


def make_password(length: int = 20) -> str:
    """A password with upper and lower case letters, digits and symbols, from ``secrets``."""
    alphabet = string.ascii_letters + string.digits + _SYMBOLS
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(char.islower() for char in password)
            and any(char.isupper() for char in password)
            and any(char.isdigit() for char in password)
            and any(char in _SYMBOLS for char in password)
        ):
            return password


def _elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'' if hours == 1 else 's'}")
    if minutes:
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")
    if secs or not parts:
        parts.append(f"{secs} second{'' if secs == 1 else 's'}")
    return " and ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]} and {parts[2]}"


class Generators:
    """Passwords, chance and a stopwatch. The stopwatch lives in memory."""

    def __init__(
        self,
        *,
        write: Callable[[str], Awaitable[Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: Any = None,
    ) -> None:
        self._write = write
        self._clock = clock
        self._rng = rng or secrets.SystemRandom()
        self._stopwatch: float | None = None

    async def answer_async(self, text: str) -> str | None:
        normalized = _normalize(text)
        password = PASSWORD_REQUEST.match(normalized)
        if password:
            length = int(password.group("length") or 20)
            if not 12 <= length <= 64:
                return "A password can be 12 to 64 characters long."
            if self._write is None:
                return None
            await self._write(make_password(length))
            return (
                f"I put a new {length}-character password on the clipboard. Paste it where you need it; "
                "I didn't keep a copy, and it isn't in your clipboard history."
            )
        if COIN_REQUEST.match(normalized):
            return f"It's {self._rng.choice(['heads', 'tails'])}."
        die = DIE_REQUEST.match(normalized)
        if die:
            if die.group("count"):
                rolls = [
                    self._rng.randint(1, 6)
                    for _ in range(_COUNTS.get(die.group("count"), 0) or int(die.group("count")))
                ]
                return f"You rolled {', '.join(map(str, rolls))}: {sum(rolls)} in total."
            sides = int(die.group("sides") or 6)
            if not 2 <= sides <= 1000:
                return "A die needs 2 to 1,000 sides."
            return f"You rolled a {self._rng.randint(1, sides)}."
        number = NUMBER_REQUEST.match(normalized)
        if number:
            low, high = sorted((int(number.group("low")), int(number.group("high"))))
            return f"{self._rng.randint(low, high)}."
        choice = CHOICE_REQUEST.match(" ".join(re.sub(r"[^\w\s,-]", " ", text.casefold()).split()))
        if choice:
            options = [part.strip() for part in re.split(r",| or ", choice.group("options")) if part.strip()]
            options = [re.sub(r"^(?:and |or )", "", option) for option in options]
            if 2 <= len(options) <= 10 and all(len(option) <= 60 for option in options):
                return f"I pick {self._rng.choice(options)}."
            return None
        if STOPWATCH_START.match(normalized):
            restarted = self._stopwatch is not None
            self._stopwatch = self._clock()
            return "Stopwatch restarted." if restarted else "Stopwatch started."
        if STOPWATCH_CHECK.match(normalized):
            if self._stopwatch is None:
                return 'The stopwatch isn\'t running. Say "start a stopwatch".'
            return f"{_elapsed(self._clock() - self._stopwatch)} so far."
        if STOPWATCH_STOP.match(normalized):
            if self._stopwatch is None:
                return "The stopwatch isn't running."
            elapsed = self._clock() - self._stopwatch
            self._stopwatch = None
            return f"Stopped at {_elapsed(elapsed)}."
        return None
