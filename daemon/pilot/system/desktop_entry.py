"""Launching a Linux desktop application the way the desktop does, from its desktop entry. Pure, tested everywhere.

Follows the freedesktop.org Desktop Entry Specification (version 1.5): the ``Exec`` key is split into arguments
with its quoting rules (double quotes; inside them a backslash escapes ``"``, `` ` ``, ``$`` and ``\\``), and its
field codes are expanded:

- ``%f`` one file, ``%F`` a list of files, ``%u`` one URL, ``%U`` a list of URLs: replaced by the arguments
  given (``%f`` and ``%u`` take the first one only, as the specification allows a launcher to start one instance
  per file; Heliox passes one);
- ``%i`` becomes ``--icon <Icon>`` when the entry has an icon, ``%c`` the entry's name, ``%k`` the entry's path;
- ``%%`` is a literal percent sign; the deprecated ``%d %D %n %N %v %m`` are removed.

When the person gives arguments and ``Exec`` has no file or URL field code, they are appended, as on Windows and
macOS (``open --args``); flags such as ``--new-window`` reach the program that way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pilot.system.app_names import parse_desktop_entry


class ExecError(ValueError):
    """An Exec line the specification does not allow (an unterminated quote, a reserved character unquoted)."""


FILE_CODES = ("%f", "%F", "%u", "%U")
DEPRECATED = ("%d", "%D", "%n", "%N", "%v", "%m")


def split_exec(line: str) -> list[str]:
    """Split an Exec value into arguments with the specification's quoting rules."""
    args: list[str] = []
    current: list[str] = []
    in_quotes = False
    has_token = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == "\\" and i + 1 < len(line) and line[i + 1] in '"`$\\':
                current.append(line[i + 1])
                i += 2
                continue
            if ch == '"':
                in_quotes = False
            else:
                current.append(ch)
        elif ch == '"':
            in_quotes = True
            has_token = True
        elif ch in " \t":
            if has_token or current:
                args.append("".join(current))
                current, has_token = [], False
        else:
            current.append(ch)
        i += 1
    if in_quotes:
        raise ExecError("unterminated quote in Exec")
    if has_token or current:
        args.append("".join(current))
    return args


def expand_exec(entry: dict[str, str], arguments: Iterable[str] = (), path: str = "") -> list[str]:
    """The argv that launches ``entry`` (the keys of its [Desktop Entry] group) with ``arguments``."""
    given = list(arguments)
    argv: list[str] = []
    used = False
    for token in split_exec(entry.get("Exec", "")):
        if token in ("%f", "%u"):
            argv.extend(given[:1])
            used = True
        elif token in ("%F", "%U"):
            argv.extend(given)
            used = True
        elif token == "%i":
            if entry.get("Icon"):
                argv.extend(["--icon", entry["Icon"]])
        elif token in DEPRECATED:
            continue
        else:
            argv.append(_expand_inline(token, entry, path))
    if not argv:
        raise ExecError("empty Exec")
    if given and not used:
        argv.extend(given)
    return argv


def _expand_inline(token: str, entry: dict[str, str], path: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(token):
        if token[i] == "%" and i + 1 < len(token):
            code = token[i + 1]
            if code == "%":
                out.append("%")
            elif code == "c":
                out.append(entry.get("Name", ""))
            elif code == "k":
                out.append(path)
            # any other code inside a word is dropped, as the specification says field codes stand alone
            i += 2
            continue
        out.append(token[i])
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class DesktopApp:
    desktop_id: str  # the file name without .desktop
    name: str
    entry: dict[str, str]
    hidden: bool

    @property
    def terminal(self) -> bool:
        return self.entry.get("Terminal", "").lower() == "true"


def desktop_apps(entries: Iterable[tuple[str, str]], which=None) -> list[DesktopApp]:
    """Launchable applications from (file stem, text) pairs. Entries whose TryExec program is missing, Hidden
    (deleted) entries, links and directories are left out. ``which`` is ``shutil.which`` (injectable for tests); None skips the TryExec check."""
    apps: list[DesktopApp] = []
    seen: set[str] = set()
    for stem, text in entries:
        if stem in seen:
            continue  # the first directory in XDG order wins, as the specification says
        seen.add(stem)
        keys = parse_desktop_entry(text)
        if keys.get("Type", "Application") != "Application" or not keys.get("Name") or not keys.get("Exec"):
            continue
        if which is not None and keys.get("TryExec") and not which(keys["TryExec"]):
            continue
        if keys.get("Hidden", "").lower() == "true":
            continue  # Hidden means deleted: the person or the system removed it
        hidden = keys.get("NoDisplay", "").lower() == "true"
        apps.append(DesktopApp(stem, keys["Name"], keys, hidden))
    return apps
