"""Where a scenario's audio lives, and how a reference names a file and a span of it (pure path logic).

A reference is a path relative to the audio root, always written with forward slashes, optionally followed by
``#start-end`` in seconds: ``real/take_2241.wav#24.0-26.5``. References may not be absolute or climb out of the
root, so a scenario file can never point at an arbitrary file on the machine. The root comes from ``--audio``,
else ``CONVERSATION_BENCH_AUDIO``, else ``~/heliox-conversation-bench-audio``. It is private and never committed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath, PureWindowsPath

ENV_VAR = "CONVERSATION_BENCH_AUDIO"
DEFAULT_DIR_NAME = "heliox-conversation-bench-audio"
_SPAN = re.compile(r"^(?P<path>[^#]+)(?:#(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?))?$")


class BadAudioRef(ValueError):
    """A reference that is malformed, absolute or outside the audio root."""


@dataclass(frozen=True)
class AudioRef:
    path: str  # relative, forward slashes
    start: float | None = None
    end: float | None = None

    @property
    def label(self) -> str:
        span = f"#{self.start:g}-{self.end:g}" if self.start is not None and self.end is not None else ""
        return f"{self.path}{span}"


def parse_ref(ref: str) -> AudioRef:
    match = _SPAN.match(ref.strip())
    if not match:
        raise BadAudioRef(f"not an audio reference: {ref!r}")
    raw = match.group("path")
    if "\\" in raw:
        raise BadAudioRef(f"use forward slashes in audio references: {ref!r}")
    posix = PurePosixPath(raw)
    if posix.is_absolute() or PureWindowsPath(raw).drive or ".." in posix.parts or not posix.parts:
        raise BadAudioRef(f"audio references must stay inside the audio folder: {ref!r}")
    start = float(match.group("start")) if match.group("start") else None
    end = float(match.group("end")) if match.group("end") else None
    if start is not None and end is not None and end <= start:
        raise BadAudioRef(f"span ends before it starts: {ref!r}")
    return AudioRef(str(posix), start, end)


def audio_root(cli_value: str | None, env: dict[str, str] | None = None, home: str | None = None) -> str:
    """The audio folder: the command-line value, else the environment variable, else a folder in home."""
    env = dict(os.environ) if env is None else env
    if cli_value:
        return cli_value
    if env.get(ENV_VAR):
        return env[ENV_VAR]
    base = home if home is not None else os.path.expanduser("~")
    return os.path.join(base, DEFAULT_DIR_NAME)


def resolve(root: str, ref: AudioRef, flavour: type[PurePath] | None = None) -> str:
    """The file a reference names under ``root``, written with the given path rules (the host's by default)."""
    rules = flavour or (PureWindowsPath if os.name == "nt" else PurePosixPath)
    return str(rules(root).joinpath(*PurePosixPath(ref.path).parts))
