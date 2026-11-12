"""Small helpers: strong passwords, coin flips, dice, random picks and a stopwatch.

"Generate a password" makes one with the operating system's secure random source and
puts it on the clipboard. It is never spoken, shown in chat, written to disk or kept in
clipboard history (which is told about each password before it is copied, and also skips
anything that looks like one). "Flip a coin",
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

from pilot.turn_changes import note_change
from pilot.workflows.clipboard_history import looks_secret

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


_ATTEMPTS = 32


def _has_every_class(password: str) -> bool:
    return (
        any(char.islower() for char in password)
        and any(char.isupper() for char in password)
        and any(char.isdigit() for char in password)
        and any(char in _SYMBOLS for char in password)
    )


def make_password(length: int = 20) -> str:
    """A password with upper and lower case letters, digits and symbols, from ``secrets``.

    Only returns passwords that clipboard history's ``looks_secret`` accepts, so anything
    that copies one around also treats it as a secret. A random draw misses that check
    (too many repeated characters) in about 1 of 300 at 12 characters, so it redraws, a
    bounded number of times. The fallback uses no repeated character, which always passes.
    """
    alphabet = string.ascii_letters + string.digits + _SYMBOLS
    for _ in range(_ATTEMPTS):
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        if _has_every_class(password) and looks_secret(password):
            return password
    rng = secrets.SystemRandom()
    picks = [rng.choice(group) for group in (string.ascii_lowercase, string.ascii_uppercase, string.digits, _SYMBOLS)]
    rest = [char for char in alphabet if char not in picks]
    picks += rng.sample(rest, length - len(picks))
    rng.shuffle(picks)
    return "".join(picks)


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
        on_password: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: Any = None,
    ) -> None:
        self._write = write
        self._on_password = on_password
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
            generated = make_password(length)
            if self._on_password is not None:
                self._on_password(generated)  # before the write, so a clipboard poll can never see it first
            await self._write(generated)
            note_change("copied it to the clipboard")
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
