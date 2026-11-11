"""Whether a path is inside a folder, under each system's rules. Pure, so both rule sets are tested everywhere.

Windows compares case-insensitively and treats / and \\ alike. macOS's default file systems (APFS and HFS+) are
case-insensitive as well, so ``~/Documents`` and ``~/documents`` are the same folder there; a protected-folder
rule that compared case-sensitively could be stepped around by changing the case of one letter. Linux file systems
are case-sensitive. A folder contains a path only at a separator boundary: ``C:\\Data`` does not contain
``C:\\Database``.
"""

from __future__ import annotations

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


def path_within(path: str, folder: str, rules: PathRules | None = None) -> bool:
    """True when ``path`` is ``folder`` or inside it. Both should already be absolute and resolved."""
    if not path or not folder:
        return False
    rules = rules or rules_for()
    sep = "\\" if rules.windows else "/"
    p, f = _key(path, rules), _key(folder, rules)
    return p == f or p.startswith(f if f.endswith(sep) else f + sep)
