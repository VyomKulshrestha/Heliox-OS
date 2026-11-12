"""Whether a path is inside a folder, under each system's rules. Pure, so both rule sets are tested everywhere.

Windows compares case-insensitively and treats / and \\ alike. macOS's default file systems (APFS and HFS+) are
case-insensitive as well, so ``~/Documents`` and ``~/documents`` are the same folder there; a protected-folder
rule that compared case-sensitively could be stepped around by changing the case of one letter. Linux file systems
are case-sensitive. A folder contains a path only at a separator boundary: ``C:\\Data`` does not contain
``C:\\Database``.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
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


def names_a_place(path: str, rules: PathRules | None = None) -> bool:
    """Whether ``path`` names a place on disk without depending on a working folder the reader cannot see, under
    ``rules`` (this system's by default): an absolute path or "~"; on Windows also a rooted path without a drive
    ("\\Users\\sam", "/Users/sam"), a drive-relative one ("C:folder") and a UNC path ("\\\\server\\share"). A bare
    name ("notes.txt", "git") is not, and on macOS and Linux neither is a path written the Windows way ("C:/Windows"
    is a relative name there, and "\\" an ordinary character).

    Decided by the text, the same on every Python Heliox supports: Python 3.13 on Windows no longer counts a rooted
    path without a drive as absolute (os.path.isabs("/etc") is False there, True on 3.11 and 3.12), yet Windows
    opens it on the current drive, so a check that asked os.path.isabs saw nothing there."""
    windows = (rules or rules_for()).windows
    text = path.strip()
    if not text:
        return False
    if text.startswith(("~", "/")):
        return True
    return windows and (text.startswith("\\") or (len(text) >= 2 and text[1] == ":" and text[0].isalpha()))


def resolved(path: str, rules: PathRules | None = None, *, cwd: str | None = None, home: str | None = None) -> str:
    """``path`` as the system will open it: "~" expanded, made absolute, links followed. On Windows a rooted path
    without a drive lands on the current drive and "C:folder" on drive C's current folder, which is where a file
    action run by the daemon opens them.

    With this system's rules and no ``cwd`` it asks the system (os.path.realpath). With another system's rules, or a
    ``cwd`` given, it works on the text alone with that system's path module (ntpath or posixpath), from ``cwd`` and
    ``home``, without following links, so each system's answer can be checked on any machine."""
    text = path.strip()
    if rules is None or (rules == rules_for() and cwd is None and home is None):
        try:
            return os.path.realpath(os.path.expanduser(text))
        except (OSError, ValueError):
            return path
    flavour = ntpath if rules.windows else posixpath
    if home is not None and (text == "~" or text.startswith(("~/", "~" + flavour.sep))):
        text = home + text[1:]
    return flavour.normpath(flavour.join(cwd or flavour.sep, text))


def path_within(path: str, folder: str, rules: PathRules | None = None) -> bool:
    """True when ``path`` is ``folder`` or inside it. Both should already be absolute and resolved."""
    if not path or not folder:
        return False
    rules = rules or rules_for()
    sep = "\\" if rules.windows else "/"
    p, f = _key(path, rules), _key(folder, rules)
    return p == f or p.startswith(f if f.endswith(sep) else f + sep)
