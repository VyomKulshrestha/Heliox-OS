"""A secret kept in a file of its own, for when the system keyring cannot be used (for example Linux without a secret
service). Used for the online learner's snapshot key, so a machine without a keyring does not replay the whole
experience ledger at every start.

The file lives in the runtime folder beside the auth tokens and is written owner-only (0600 on Linux and macOS; on
Windows the runtime folder is in the person's own profile, whose permissions already keep other users out). A file that
other users can read or write is refused, not used and not rewritten, and the reason is said in plain words.

What it protects against: a damaged or truncated snapshot, a snapshot from another version, and other users of the
computer. What it does not: anyone who can already read or write files as this user, who could read this key as
easily as the snapshot itself.
"""

from __future__ import annotations

import os
import secrets
import stat
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


def load_or_create(path: Path) -> bytes:
    """The key in ``path``, made (owner-only) on first use. Raises KeyFileRefused for a file others can use."""
    if path.exists():
        _check_owner_only(path)
        secret = path.read_text(encoding="utf-8").strip()
        if len(secret) >= 32:
            return secret.encode("utf-8")
        raise KeyFileRefused(
            f"The key file {path} is damaged (too short), so it was not used. Delete it to start over."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)  # owner-only from the first byte, not chmod after writing
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secret)
    return secret.encode("utf-8")
