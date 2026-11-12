"""Running a program a plan names on Windows, so what the person approved is what runs.

cmd.exe reads its whole command line again before running it: & | < > ^ start another command or redirect, % and
! expand variables, ( ) group, and " changes what the rest means. A plan's shell_command used to go through
"cmd /c" with every argument, so an argument holding one of those (a file name from the plan, text from the
screen) ran a second command the approval never showed as one. Now:

- a program found on PATH or by path (an .exe or .com) runs directly, with its arguments passed as they are;
- a cmd built-in (dir, type, echo and the others in ``CMD_BUILTINS``) and a .bat or .cmd file still need cmd, so
  they go through "cmd /c", and only when neither the command nor any argument holds a character cmd would read
  as code; otherwise the action is refused with a sentence that says which character and why. (A batch file is
  run by cmd however it is started, so it gets the same check.)
"""

from __future__ import annotations

import os
import shutil

# cmd.exe's built-in commands: there is no program to run directly.
CMD_BUILTINS = frozenset(
    {
        "assoc",
        "break",
        "call",
        "cd",
        "chdir",
        "cls",
        "color",
        "copy",
        "date",
        "del",
        "dir",
        "echo",
        "endlocal",
        "erase",
        "ftype",
        "md",
        "mkdir",
        "mklink",
        "move",
        "path",
        "popd",
        "prompt",
        "pushd",
        "rd",
        "ren",
        "rename",
        "rmdir",
        "set",
        "setlocal",
        "start",
        "time",
        "title",
        "type",
        "ver",
        "verify",
        "vol",
    }
)
# Characters cmd reads as code in a command line: command separators and redirection, its escape, variable
# expansion, grouping, quoting, and line breaks.
CMD_SPECIAL = frozenset('&|<>^%!()"\r\n')


class UnsafeCommandLine(ValueError):
    """An argument cmd would read as code; the message is the sentence the person sees."""


def needs_cmd(command: str) -> bool:
    """Whether ``command`` only runs through cmd: a built-in, or a batch file."""
    name = command.strip().strip('"')
    if name.casefold() in CMD_BUILTINS:
        return True
    resolved = shutil.which(name) or name
    return os.path.splitext(resolved)[1].casefold() in (".bat", ".cmd")


def windows_command_line(command: str, args: list[str]) -> list[str]:
    """The argument list to run ``command`` with ``args`` on Windows: directly, or through cmd /c when it must and
    nothing in it would be read as code. Raises ``UnsafeCommandLine`` otherwise.

    A plan sometimes gives a whole line as the command ("dir /b C:\\\\notes") with no arguments, which "cmd /c" used to
    take apart; it is split here the way Windows splits a command line (quotes keep a path with spaces together)."""
    if not args and any(char.isspace() for char in command.strip()):
        import shlex

        parts = [part.strip('"') for part in shlex.split(command.strip(), posix=False)]
        command, args = parts[0], parts[1:]
    if not needs_cmd(command):
        return [command, *args]
    for value in (command, *args):
        found = sorted({char for char in value if char in CMD_SPECIAL})
        if found:
            shown = ", ".join(repr(char) for char in found)
            raise UnsafeCommandLine(
                f"Not run: {command!r} goes through cmd, which would read {shown} in {value!r} as part of the "
                "command, not as text, so what ran could differ from what was approved. Use a program that runs "
                "directly, or a shell_script with interpreter powershell."
            )
    return ["cmd", "/c", command, *args]


def batch_file_command(script_path: str) -> list[str]:
    """cmd /c running the daemon's own temporary batch file. cmd reads the path as part of its command line too, so
    a temporary folder whose path holds & ^ % or a quote (it holds the user name) is refused, not run."""
    found = sorted({char for char in script_path if char in CMD_SPECIAL})
    if found:
        raise UnsafeCommandLine(
            f"Not run: the temporary file's path {script_path!r} holds {', '.join(map(repr, found))}, which cmd "
            "would read as part of the command."
        )
    return ["cmd", "/c", script_path]


async def open_with_system(target: str) -> tuple[int, str]:
    """Open a web address or file with the system's handler for it (ShellExecute on Windows), with no command
    line in between. (exit code, error text)."""
    import asyncio

    try:
        await asyncio.to_thread(os.startfile, target)  # type: ignore[attr-defined]
    except OSError as error:
        return 1, str(error)
    return 0, ""
