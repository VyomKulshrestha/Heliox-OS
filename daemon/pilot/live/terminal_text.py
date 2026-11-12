"""The command line being typed in kitty or WezTerm, read through the terminal's own command-line control.

Both terminals show nothing to accessibility on Linux (and on macOS only Terminal.app and iTerm2 do), so a risky
command typed there reached Live only after it ran. For people who would rather not add the zsh or fish hook
(``pilot.live.shell_line``), Live asks the terminal itself:

* **WezTerm**: ``wezterm cli list-clients`` names the pane that has focus, ``wezterm cli list`` gives that pane's
  cursor row, and ``wezterm cli get-text --pane-id N`` its screen. Nothing to set up.
* **kitty**: ``kitty @ ls`` finds the focused window and ``kitty @ get-text --extent screen --add-cursor`` gives its
  screen with the cursor position. kitty answers only with remote control on (``allow_remote_control
  socket-only`` and ``listen_on unix:/tmp/kitty`` in kitty.conf); Settings says so, and Heliox never changes
  kitty.conf.

Only the line the cursor is on is kept, and from it only what follows the prompt (``$``, ``%``, ``#``, ``>``,
``❯``, ``➜``). The rules of the shell hook apply: never a line that starts with a space, never anything from the
first key-looking string onwards (``shell_line.readable``). Asked only while kitty or WezTerm is the window in
front, Live is on and reading fields is on (the caller checks), at most every ``EVERY_SECONDS``, in the background
so Live's tick never waits for it; each call is killed after ``CALL_SECONDS``. The line stays in memory for
``FRESH_SECONDS`` and is never sent to a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot.live.shell_line import readable

logger = logging.getLogger("pilot.live.terminal_text")

EVERY_SECONDS = 0.6  # Live's own perception pace
CALL_SECONDS = 1.0  # a terminal that does not answer within this is given up on, for this look
FRESH_SECONDS = 2.0
TERMINALS = ("kitty", "wezterm")
_PROMPT = re.compile(r"(?:^|[\s\\:)\]])(?:[$%#>❯➜λ»])\s")  # "user@host ~ $ ", "PS C:\> ", "[main]> "
_CURSOR = re.compile(r"\x1b\[(\d+);(\d+)H")
_ANSI = re.compile(r"\x1b\[[0-9;? ]*[A-Za-z]|\x1b\][^\x07]*\x07")  # colours, cursor moves and shape ("\x1b[2 q")

Runner = Callable[[list[str], float], Awaitable[tuple[int | None, str]]]


def which_terminal(app: str, title: str = "") -> str:
    """ "kitty", "wezterm" or "" for the app in front (``wezterm-gui``, ``WezTerm``, ``kitty``, ``kitty.app``)."""
    name = app.casefold()
    if "wezterm" in name:
        return "wezterm"
    if name.removesuffix(".app") == "kitty" or name.startswith("kitty"):
        return "kitty"
    return ""


def command_part(line: str) -> str:
    """What follows the last prompt sign on a screen line ("vk@box ~/src $ git push -f" -> "git push -f")."""
    found = list(_PROMPT.finditer(line))
    return line[found[-1].end() :] if found else line


def typed_line(screen_line: str) -> str | None:
    """The part of the cursor's line Live may read, or None."""
    text = _ANSI.sub("", screen_line).rstrip()
    return readable(command_part(text)) if text.strip() else None


# ── WezTerm (pure) ──


def wezterm_focused_pane(clients_json: str) -> int | None:
    try:
        clients = json.loads(clients_json)
    except ValueError:
        return None
    # The client most recently active is the one the person is using.
    best = None
    for client in clients if isinstance(clients, list) else []:
        pane = client.get("focused_pane_id")
        idle = client.get("idle_time", {})
        seconds = idle.get("secs", 0) + idle.get("nanos", 0) / 1e9 if isinstance(idle, dict) else 0
        if isinstance(pane, int) and (best is None or seconds < best[0]):
            best = (seconds, pane)
    return best[1] if best else None


def wezterm_cursor_row(list_json: str, pane_id: int) -> int | None:
    """The cursor's row in the pane's visible screen (0 is the top line)."""
    try:
        panes = json.loads(list_json)
    except ValueError:
        return None
    for pane in panes if isinstance(panes, list) else []:
        if pane.get("pane_id") == pane_id:
            row = pane.get("cursor_y")
            return row if isinstance(row, int) else None
    return None


def screen_row(text: str, row: int | None) -> str:
    lines = text.split("\n")
    if row is not None and 0 <= row < len(lines):
        return lines[row]
    rest = [line for line in lines if line.strip()]
    return rest[-1] if rest else ""


# ── kitty (pure) ──


def kitty_socket_setting(conf: str) -> tuple[bool, str]:
    """(remote control allowed for the socket, ``listen_on`` value) from kitty.conf's text."""
    allowed, listen = False, ""
    for raw in conf.splitlines():
        parts = raw.strip().split(None, 1)
        if len(parts) != 2 or parts[0].startswith("#"):
            continue
        key, value = parts[0], parts[1].strip()
        if key == "allow_remote_control":
            allowed = value.casefold() in ("yes", "y", "true", "socket-only")  # "password" needs one Heliox lacks
        elif key == "listen_on":
            listen = value
    return allowed, listen


def kitty_addresses(listen: str, pid: int | None) -> list[str]:
    """Where a kitty with this ``listen_on`` answers: kitty adds ``-<its pid>`` unless ``{kitty_pid}`` is used."""
    if not listen.startswith("unix:"):
        return []
    if "{kitty_pid}" in listen:
        return [listen.replace("{kitty_pid}", str(pid))] if pid else []
    return ([f"{listen}-{pid}"] if pid else []) + [listen]


def kitty_focused_window(ls_json: str) -> int | None:
    try:
        tree = json.loads(ls_json)
    except ValueError:
        return None
    # kitty is the app in front (the caller checked), so when no OS window says it has focus (no window manager,
    # or focus not reported yet) the last active one is the one in front.
    for flag in ("is_focused", "is_active"):
        for os_window in tree if isinstance(tree, list) else []:
            if not os_window.get(flag):
                continue
            for tab in os_window.get("tabs", []):
                if not tab.get(flag):
                    continue
                for window in tab.get("windows", []):
                    if window.get(flag):
                        return window.get("id")
    return None


def kitty_cursor_line(text: str) -> str:
    """With ``--add-cursor`` kitty ends the text with the cursor's position (1-based row;col)."""
    found = list(_CURSOR.finditer(text))
    row = int(found[-1].group(1)) - 1 if found else None
    return screen_row(_ANSI.sub("", text), row)


# ── finding the programs ──


def _program(name: str, which: Callable[[str], str | None] = shutil.which) -> str | None:
    found = which(name)
    if found:
        return found
    if sys.platform == "darwin":
        bundle = {"kitty": "/Applications/kitty.app/Contents/MacOS/kitty"}.get(
            name, "/Applications/WezTerm.app/Contents/MacOS/wezterm"
        )
        if Path(bundle).exists():
            return bundle
    return None


def kitty_pids() -> list[int]:
    """Running kitty processes, newest first (kitty shows nothing to accessibility, so its pid comes from here)."""
    try:
        import psutil

        found = [(p.info.get("create_time") or 0.0, p.info["pid"]) for p in psutil.process_iter(["name", "pid", "create_time"])
                 if (p.info.get("name") or "").casefold() in ("kitty", "kitty.exe")]  # fmt: skip
    except Exception:
        return []
    return [pid for _, pid in sorted(found, reverse=True)]


def kitty_conf_path(env: dict[str, str], home: Path) -> Path:
    if env.get("KITTY_CONFIG_DIRECTORY"):
        return Path(env["KITTY_CONFIG_DIRECTORY"]) / "kitty.conf"
    return Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "kitty" / "kitty.conf"


async def run(argv: list[str], timeout: float) -> tuple[int | None, str]:
    """(exit code, stdout), or (None, "") when it had to be killed or could not start."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except OSError:
        return None, ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return None, ""
    return proc.returncode, out.decode("utf-8", errors="replace")


# ── the reader ──


@dataclass
class Seen:
    at: float
    terminal: str
    line: str
    pane: str  # which pane or window, stable while the person types in it


class TerminalText:
    """The latest typed line from kitty or WezTerm, refreshed in the background."""

    def __init__(
        self,
        *,
        runner: Runner = run,
        which: Callable[[str], str | None] = shutil.which,
        env: dict[str, str] | None = None,
        home: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run = runner
        self._which = which
        self._env = dict(os.environ) if env is None else env
        self._home = home or Path.home()
        self._clock = clock
        self._seen: Seen | None = None
        self._asked_at = -1e9
        self._task: asyncio.Task[None] | None = None
        self.problem: dict[str, str] = {}  # per terminal, why it could not be read, for Settings

    def latest(self, terminal: str) -> Seen | None:
        seen = self._seen
        if seen is None or seen.terminal != terminal or self._clock() - seen.at > FRESH_SECONDS:
            return None
        return seen

    def poke(self, terminal: str, pid: int | None = None) -> None:
        """Ask the terminal again in the background, unless asked a moment ago or still asking."""
        now = self._clock()
        if terminal not in TERMINALS or now - self._asked_at < EVERY_SECONDS:
            return
        if self._task is not None and not self._task.done():
            return
        self._asked_at = now
        with contextlib.suppress(RuntimeError):
            self._task = asyncio.get_running_loop().create_task(self.refresh(terminal, pid))

    async def refresh(self, terminal: str, pid: int | None = None) -> Seen | None:
        try:
            found = await (self._wezterm() if terminal == "wezterm" else self._kitty(pid))
        except Exception:
            logger.debug("Reading %s failed", terminal, exc_info=True)
            found = None
        if found is None:
            if self._seen is not None and self._seen.terminal == terminal:
                self._seen = None
            return None
        pane, screen_line = found
        line = typed_line(screen_line)
        self._seen = Seen(self._clock(), terminal, line, pane) if line else None
        return self._seen

    async def _wezterm(self) -> tuple[str, str] | None:
        exe = _program("wezterm", self._which)
        if not exe:
            self.problem["wezterm"] = "The wezterm command is not on this computer's PATH."
            return None
        code, clients = await self._run([exe, "cli", "list-clients", "--format", "json"], CALL_SECONDS)
        pane = wezterm_focused_pane(clients) if code == 0 else None
        if pane is None:
            self.problem["wezterm"] = "WezTerm did not say which pane has focus."
            return None
        code, panes = await self._run([exe, "cli", "list", "--format", "json"], CALL_SECONDS)
        row = wezterm_cursor_row(panes, pane) if code == 0 else None
        code, text = await self._run([exe, "cli", "get-text", "--pane-id", str(pane)], CALL_SECONDS)
        if code != 0:
            self.problem["wezterm"] = "WezTerm did not give the pane's text."
            return None
        self.problem.pop("wezterm", None)
        return f"wezterm:{pane}", screen_row(text, row)

    def kitty_setting(self) -> tuple[bool, str]:
        path = kitty_conf_path(self._env, self._home)
        try:
            return kitty_socket_setting(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return False, ""

    async def _kitty(self, pid: int | None) -> tuple[str, str] | None:
        exe = _program("kitty", self._which)
        if not exe:
            self.problem["kitty"] = "The kitty command is not on this computer's PATH."
            return None
        allowed, listen = self.kitty_setting()
        if not allowed or not listen.startswith("unix:"):
            self.problem["kitty"] = KITTY_SETUP
            return None
        pids = [pid] if pid else kitty_pids()
        addresses = [a for one in (pids or [None]) for a in kitty_addresses(listen, one)]
        for address in dict.fromkeys(addresses):
            code, tree = await self._run([exe, "@", "--to", address, "ls"], CALL_SECONDS)
            if code != 0:
                continue
            window = kitty_focused_window(tree)
            if window is None:
                continue  # another kitty, not the one in front
            code, text = await self._run(
                [
                    exe,
                    "@",
                    "--to",
                    address,
                    "get-text",
                    "--match",
                    f"id:{window}",
                    "--extent",
                    "screen",
                    "--add-cursor",
                ],
                CALL_SECONDS,
            )
            if code != 0:
                break
            self.problem.pop("kitty", None)
            return f"kitty:{address}:{window}", kitty_cursor_line(text)
        self.problem["kitty"] = "kitty did not answer on its remote-control socket. " + KITTY_SETUP
        return None

    def status(self) -> dict[str, Any]:
        """For Settings: what each terminal needs, and why the last look failed."""
        allowed, listen = self.kitty_setting()
        return {
            "wezterm": {
                "found": bool(_program("wezterm", self._which)),
                "needs": "",
                "problem": self.problem.get("wezterm", ""),
            },
            "kitty": {
                "found": bool(_program("kitty", self._which)),
                "ready": allowed and listen.startswith("unix:"),
                "needs": "" if allowed and listen.startswith("unix:") else KITTY_SETUP,
                "problem": self.problem.get("kitty", ""),
            },
        }


KITTY_SETUP = (
    "kitty answers only with remote control on: add 'allow_remote_control socket-only' and "
    "'listen_on unix:/tmp/kitty' to kitty.conf, then restart kitty."
)

__all__ = [
    "KITTY_SETUP",
    "TERMINALS",
    "Seen",
    "TerminalText",
    "command_part",
    "kitty_addresses",
    "kitty_cursor_line",
    "kitty_focused_window",
    "kitty_socket_setting",
    "screen_row",
    "typed_line",
    "which_terminal",
    "wezterm_cursor_row",
    "wezterm_focused_pane",
]
