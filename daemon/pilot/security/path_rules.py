"""Whether a path is inside a folder, under each system's rules. Pure, so both rule sets are tested everywhere.

Windows compares case-insensitively and treats / and \\ alike. macOS's default file systems (APFS and HFS+) are
case-insensitive as well, so ``~/Documents`` and ``~/documents`` are the same folder there; a protected-folder
rule that compared case-sensitively could be stepped around by changing the case of one letter. Linux file systems
are case-sensitive. A folder contains a path only at a separator boundary: ``C:\\Data`` does not contain
``C:\\Database``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PathRules:
    case_insensitive: bool
    windows: bool


WINDOWS = PathRules(case_insensitive=True, windows=True)
MACOS = PathRules(case_insensitive=True, windows=False)
LINUX = PathRules(case_insensitive=False, windows=False)


def rules_for(platform: str = sys.platform) -> PathRules:
    if platform == "win32":
        return WINDOWS
    if platform == "darwin":
        return MACOS
    return LINUX


def _key(path: str, rules: PathRules) -> str:
    if rules.windows:
        path = path.replace("/", "\\")
        sep = "\\"
    else:
        sep = "/"
    if len(path) > 1:
        path = path.rstrip(sep) or sep
        if rules.windows and len(path) == 2 and path[1] == ":":
            path += sep  # "C:" is the drive root
    return path.casefold() if rules.case_insensitive else path


def names_a_place(path: str, windows: bool | None = None) -> bool:
    """Whether ``path`` names a place on disk without depending on a working folder the reader cannot see: an
    absolute path, "~", and on Windows also a rooted path without a drive ("\\Users\\sam", "/Users/sam"), a
    drive-relative one ("C:folder") and a UNC path ("\\\\server\\share"). A bare name ("notes.txt", "git") is not.

    Decided by the text, the same on every Python Heliox supports: Python 3.13 on Windows no longer counts a rooted
    path without a drive as absolute (os.path.isabs("/etc") is False there, True on 3.11 and 3.12), yet Windows
    opens it on the current drive, so a check that asked os.path.isabs saw nothing there."""
    windows = sys.platform == "win32" if windows is None else windows
    text = path.strip()
    if not text:
        return False
    if text.startswith(("~", "/")):
        return True
    return windows and (text.startswith("\\") or (len(text) >= 2 and text[1] == ":" and text[0].isalpha()))


def resolved(path: str) -> str:
    """``path`` as the system will open it: "~" expanded, then made absolute and links followed. On Windows a rooted
    path without a drive lands on the daemon's current drive and "C:folder" on drive C's current folder, which is
    where a file action run by the daemon opens them."""
    try:
        return os.path.realpath(os.path.expanduser(path.strip()))
    except (OSError, ValueError):
        return path


def path_within(path: str, folder: str, rules: PathRules | None = None) -> bool:
    """True when ``path`` is ``folder`` or inside it. Both should already be absolute and resolved."""
    if not path or not folder:
        return False
    rules = rules or rules_for()
    sep = "\\" if rules.windows else "/"
    p, f = _key(path, rules), _key(folder, rules)
    return p == f or p.startswith(f if f.endswith(sep) else f + sep)
