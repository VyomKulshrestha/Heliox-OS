"""Values handed to PowerShell and AppleScript as data, never pasted into the code they run.

A value written into script text, even escaped, can end the string it sits in: PowerShell takes the typographic
quotes U+2018 to U+201B as single quotes too, and one missed character turns a file name, a window title or an
invitation's title into code. So no value from a plan, the screen, a file name or a message is ever put into the
script text. The script reads it instead:

- PowerShell (``powershell_values``, and ``run_powershell(script, values=...)`` in platform_detect): each value
  is in ``$HelioxArgs['name']``. A value up to ``ENV_VALUE_LIMIT`` characters travels in an environment variable
  of that one process (``HELIOX_ARG_NAME``); a longer one (spoken text, clipboard text, a long message: a Windows
  environment variable holds about 32,000 characters and the whole block has a limit too) in a temporary file
  only its owner can read, named by ``HELIOX_ARGFILE_NAME`` and removed when the command ends, whatever happens.
  The script text then holds only the fixed prelude and the caller's constant code; names are checked against a
  pattern, so they cannot carry code either.
- AppleScript (``osascript_command``): the values are osascript's arguments, which reach the script's
  ``on run argv`` handler as a list (``item 1 of heliox_args``). Each argument is given a leading "x" that the
  prelude removes, so a value beginning with "-" is never taken as an option of osascript.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import Iterator, Mapping

ARG_ENV_PREFIX = "HELIOX_ARG_"
FILE_ENV_PREFIX = "HELIOX_ARGFILE_"
ENV_VALUE_LIMIT = 4000  # characters; longer values go through an owner-only file
POWERSHELL = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _checked(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"A script value name must match {_NAME.pattern}: {name!r}")
    return name


def _owner_only_file(value: str) -> str:
    """A new file holding ``value`` as UTF-8, readable by its owner only (mkstemp opens it 0600 on macOS and Linux;
    on Windows it is in the person's own temp folder, which other users cannot read)."""
    fd, path = tempfile.mkstemp(prefix="heliox-arg-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
    return path


@contextlib.contextmanager
def powershell_values(script: str, values: Mapping[str, object]) -> Iterator[tuple[list[str], dict[str, str]]]:
    """The command line and environment that run ``script`` with ``values`` readable as ``$HelioxArgs['name']``.
    Files made for long values are removed when the ``with`` block ends."""
    env = dict(os.environ)
    files: list[str] = []
    prelude = ["$HelioxArgs = @{}"]
    try:
        for name, raw in values.items():
            key = _checked(name)
            value = "" if raw is None else str(raw)
            upper = key.upper()
            if len(value) <= ENV_VALUE_LIMIT and "\x00" not in value:
                env[ARG_ENV_PREFIX + upper] = value
                prelude.append(
                    f"$HelioxArgs['{key}'] = [Environment]::GetEnvironmentVariable('{ARG_ENV_PREFIX}{upper}')"
                )
            else:
                path = _owner_only_file(value)
                files.append(path)
                env[FILE_ENV_PREFIX + upper] = path
                prelude.append(
                    f"$HelioxArgs['{key}'] = [IO.File]::ReadAllText("
                    f"[Environment]::GetEnvironmentVariable('{FILE_ENV_PREFIX}{upper}'), [Text.Encoding]::UTF8)"
                )
            # An empty environment variable reads back as $null in PowerShell; the prelude makes it "".
            prelude.append(f"if ($HelioxArgs['{key}'] -eq $null) {{ $HelioxArgs['{key}'] = '' }}")
        yield [*POWERSHELL, "; ".join(prelude) + "; " + script], env
    finally:
        for path in files:
            with contextlib.suppress(OSError):
                os.unlink(path)


_OSA_PRELUDE = """on run argv
set heliox_args to {}
repeat with heliox_item in argv
set heliox_text to (heliox_item as text)
if (count of heliox_text) > 1 then
set end of heliox_args to (text 2 thru -1 of heliox_text)
else
set end of heliox_args to ""
end if
end repeat
"""


def osascript_command(body: str, values: list[str] | tuple[str, ...] = ()) -> list[str]:
    """``osascript`` running ``body`` (constant AppleScript) with ``values`` as ``item N of heliox_args``."""
    script = _OSA_PRELUDE + body.strip() + "\nend run"
    return ["osascript", "-e", script, *("x" + str(value) for value in values)]
