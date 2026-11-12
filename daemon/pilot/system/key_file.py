"""A secret kept in a file of its own: the online learner's snapshot key, so a start does not replay the whole
experience ledger.

The file lives in a folder of its own inside Heliox's data folder (``keys``, beside the snapshot it signs), because
that survives a reboot on every system and removal already knows it as data. It is owner-only: the folder 0700 and
the file 0600 on Linux and macOS, written that way from the first byte; on Windows the data folder is in the person's
own profile, whose permissions already keep other users out. A file that other users can read or write is refused,
not used and not rewritten, and the reason is said in plain words.

Earlier versions kept the key in the runtime folder, which Linux empties at every reboot and macOS keeps under
Library/Caches, which the system may clear. ``move_key`` moves such a key to the new place on start.

What the key protects against: a snapshot that is truncated or corrupted, one from another version, and one copied on
its own from another machine or from a backup made under another key. What it does not: a program already running
as the same user, which can read the key as easily as the snapshot (the restricted unpickler is the defence there),
and a copy of the whole data folder, which carries the key with it.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path


class KeyFileRefused(Exception):
    """Why the key file was not used, in plain words."""


def _check_owner_only(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KeyFileRefused(
            f"The key file {path} can be read or changed by other users of this computer (permissions "
            f"{stat.filemode(mode)}), so it was not used. Make it private with: chmod 600 {path}"
        )


def _private_folder(folder: Path) -> None:
    """The key's own folder, owner-only (0700 on Linux and macOS). It holds keys and nothing else, so a wider mode
    is narrowed."""
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix" and stat.S_IMODE(folder.stat().st_mode) != 0o700:
        os.chmod(folder, 0o700)


def _read(path: Path) -> str:
    _check_owner_only(path)
    secret = path.read_text(encoding="utf-8").strip()
    if len(secret) < 32:
        raise KeyFileRefused(
            f"The key file {path} is damaged (too short), so it was not used. Delete it to start over."
        )
    return secret


def _write_new(path: Path, secret: str) -> None:
    """Owner-only from the first byte (not chmod after writing), and on disk before anyone relies on it."""
    _private_folder(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secret)
        handle.flush()
        os.fsync(handle.fileno())


def load_or_create(path: Path) -> bytes:
    """The key in ``path``, made (owner-only, in an owner-only folder) on first use. Raises KeyFileRefused for a file
    others can use or a damaged one."""
    if path.exists():
        _private_folder(path.parent)
        return _read(path).encode("utf-8")
    secret = secrets.token_hex(32)
    _write_new(path, secret)
    return secret.encode("utf-8")


def move_key(old: Path, new: Path, signs: Callable[[bytes], bool]) -> str:
    """Move an earlier version's key from ``old`` to ``new`` when it is the key of the saved state here (``signs``),
    then remove the old file. Returns what was done, for the log; "" when there was nothing to do.

    The old key is moved only when it signs this data folder's snapshot: two data folders can share one runtime
    folder (a second home set through XDG_DATA_HOME, a test), and a key that belongs to the other must stay where it
    is. A copy left by a move that was interrupted (the same key in both places) is removed.
    """
    if not old.exists():
        return ""
    if new.exists():
        with contextlib.suppress(OSError, KeyFileRefused):
            if _read(old) == _read(new):
                old.unlink()
                return f"removed {old}, a copy of the key already moved to {new}"
        return ""
    try:
        secret = _read(old)
    except (OSError, KeyFileRefused) as refused:
        return f"left {old} where it is: {refused}"
    if not signs(secret.encode("utf-8")):
        return ""  # not the key of this folder's saved state
    _write_new(new, secret)
    old.unlink()
    return f"moved the key from {old} to {new}, so the saved state survives a reboot"
