"""Whether a requested package is a protected one, by the names each system's package manager accepts.

Names are compared ignoring case: winget ids and Homebrew names are case-insensitive, and Linux distributions name
packages in lower case. On Windows, winget takes an id ("Mozilla.Firefox") or a name ("firefox") for the same
package, so a protected id also protects the name after its last dot, and a protected name also protects any id
that ends in it. Elsewhere a dot is part of the name ("python3.12"), so names must be equal. Pure.
"""

from __future__ import annotations

import sys


def _windows(platform: str | None) -> bool:
    return (platform or sys.platform) == "win32"


def names_for(name: str, platform: str | None = None) -> set[str]:
    """The forms of ``name`` that count as the same package on ``platform`` (sys.platform by default)."""
    folded = name.strip().casefold()
    if not folded:
        return set()
    forms = {folded}
    if _windows(platform) and "." in folded:
        forms.add(folded.rsplit(".", 1)[1])
    return forms


def is_protected(name: str, protected: list[str] | tuple[str, ...], platform: str | None = None) -> str | None:
    """The protected entry ``name`` stands for, or None."""
    wanted = names_for(name, platform)
    if not wanted:
        return None
    for entry in protected:
        if wanted & names_for(str(entry), platform):
            return str(entry)
    return None
