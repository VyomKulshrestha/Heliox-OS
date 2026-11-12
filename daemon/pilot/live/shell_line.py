"""The command line being typed in a terminal that exposes nothing to accessibility, for Heliox Live.

On Windows every console is read through UI Automation. On Linux, kitty, Alacritty, WezTerm, foot and xterm expose
no text, and on macOS only Terminal.app and iTerm2 do, so a risky command typed there ("rm -rf ~/", "git push
--force") reached Live only after it had run. This is the opt-in way around it, the way prompt tools hook into a
shell:

* **zsh**: a ZLE hook (``line-pre-redraw``) sends the line being edited to Heliox each time it changes. Live checks
  it like a terminal field it reads itself, so the card for a risky command comes at the first pause, before Enter.
* **fish**: fish has no hook on every change to the line, so its Enter key asks Heliox first (at most one second,
  through ``nc``). A risky command is not run: the warning is printed under the line and Enter again runs it.
* **bash**: plain bash only offers hooks after Enter. With ble.sh it could work like zsh; that is not built yet.

Nothing is installed or added to a start-up file until the person presses Add in Settings, after seeing the exact
line (zsh) or file (fish) and how to remove it. What travels and where:

* over a Unix socket in a folder only this user can open (mode 0700, owned by them; Heliox refuses a folder anyone
  else owns or can open), never over the network;
* only while Live is on: with Live off the socket does not exist, and the hook stays silent;
* never a line that starts with a space (the shells' own "keep this out of history" habit), never a line typed in
  ``vared`` or a program's own prompt (zsh's ``CONTEXT`` must be ``start``; sudo and password prompts do not use the
  line editor at all), and never anything from the first token-looking string onwards (a key, a long random string):
  the line is cut there before Live sees it.

The line is kept in memory only, for ``FRESH_SECONDS``, and never sent to a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.live.shell_line")

PROTOCOL = "HLX1"
FRESH_SECONDS = 3.0  # a line not updated for this long is no longer what is on screen
MAX_LINE = 2000
SHELLS = ("zsh", "fish")
MARKER = "# Heliox Live terminal check"
# A long run of key-like characters: everything from here on is never read.
_TOKENISH = re.compile(r"(?<![\w/.-])[A-Za-z0-9_+/=-]{32,}|\b(?:sk|pk|ghp|gho|xox[abp]|AKIA)[-_A-Za-z0-9]{12,}")


# ── where the socket lives (pure) ──


def socket_path(env: dict[str, str], uid: int) -> Path:
    """The socket in a folder only this user can open: the session's runtime folder, else a per-user temp folder."""
    runtime = env.get("XDG_RUNTIME_DIR", "")
    if runtime:
        return Path(runtime) / "heliox" / "live-shell.sock"
    temp = env.get("TMPDIR", "") or "/tmp"
    return Path(temp) / f"heliox-{uid}" / "live-shell.sock"


def folder_problem(mode: int, owner: int, uid: int) -> str:
    """Why a socket folder may not be used, or "" when it is safe: it must be ours and closed to everyone else."""
    if owner != uid:
        return "the folder belongs to another user"
    if mode & 0o077:
        return "other users can open the folder"
    return ""


# ── the messages (pure) ──


@dataclass(frozen=True)
class Message:
    kind: str  # "line": the line changed (no reply); "check": Enter was pressed (one reply line)
    shell: str
    term: str
    session: str
    line: str


def parse(raw: str) -> Message | None:
    """``HLX1 <kind> <shell> <term> <session> <line>``; the line has backslashes and new lines escaped."""
    parts = raw.rstrip("\r\n").split(" ", 5)
    if len(parts) < 5 or parts[0] != PROTOCOL or parts[1] not in ("line", "check") or parts[2] not in SHELLS:
        return None
    line = parts[5] if len(parts) == 6 else ""
    line = re.sub(r"\\(\\|n)", lambda m: "\\" if m.group(1) == "\\" else "\n", line)
    return Message(parts[1], parts[2], parts[3][:40], parts[4][:20], line[:MAX_LINE])


def readable(line: str) -> str | None:
    """What of the line Live may read: nothing when it starts with a space, and nothing from a token onwards."""
    if not line.strip() or line[:1] in (" ", "\t"):
        return None
    match = _TOKENISH.search(line)
    if match:
        line = line[: match.start()]
    return line if line.strip() else None


def verdict(line: str, term: str) -> str:
    """The reply to fish's Enter: "ok", or "warn<TAB>title<TAB>message" for a risky command."""
    from datetime import date

    from pilot.live.detectors import TextContext, detect

    text = readable(line)
    if text is None:
        return "ok"
    ctx = TextContext(today=date.today(), app=term or "terminal", title="fish shell", source="focused_field")
    found = [c for c in detect(text, ctx, kinds=("risky_command",)) if c.urgency == "urgent"]
    if not found:
        return "ok"
    first = found[0]
    clean = [" ".join(str(part).replace("\t", " ").split()) for part in (first.title, first.message)]
    return "warn\t" + "\t".join(clean)


# ── what Live sees ──


@dataclass
class Seen:
    at: float
    shell: str
    term: str
    line: str


class ShellLines:
    """The latest line per shell session, in memory, for ``FRESH_SECONDS``."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._sessions: dict[str, Seen] = {}

    def take(self, message: Message) -> None:
        text = readable(message.line)
        key = f"{message.shell}:{message.session}"
        if text is None:
            self._sessions.pop(key, None)
            return
        self._sessions[key] = Seen(self._clock(), message.shell, message.term, text)

    def latest(self) -> tuple[str, Seen] | None:
        now = self._clock()
        self._sessions = {k: v for k, v in self._sessions.items() if now - v.at <= FRESH_SECONDS}
        if not self._sessions:
            return None
        return max(self._sessions.items(), key=lambda kv: kv[1].at)

    def clear(self) -> None:
        self._sessions.clear()


# ── the socket ──


class ShellLineServer:
    """A Unix socket that exists only while ``enabled()`` (Live on) and only on macOS and Linux."""

    def __init__(
        self,
        lines: ShellLines,
        *,
        enabled: Callable[[], bool],
        path: Path | None = None,
        uid: int | None = None,
    ) -> None:
        self.lines = lines
        self._enabled = enabled
        self.path = path or socket_path(dict(os.environ), _uid())
        self._uid = _uid() if uid is None else uid
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None
        self.problem = ""

    @property
    def listening(self) -> bool:
        return self._server is not None

    async def open(self) -> bool:
        if sys.platform == "win32" or self._server is not None:
            return self._server is not None
        folder = self.path.parent
        try:
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = folder.stat()
        except OSError as error:
            self.problem = f"could not make {folder}: {error}"
            return False
        self.problem = folder_problem(stat.S_IMODE(info.st_mode), info.st_uid, self._uid)
        if self.problem:
            logger.warning("Not opening the Live shell socket in %s: %s", folder, self.problem)
            return False
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        old = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._client, path=str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        return True

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        with contextlib.suppress(OSError):
            self.path.unlink()
        self.lines.clear()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                message = parse(raw.decode("utf-8", errors="replace"))
                if message is None or not self._enabled():
                    continue
                if message.kind == "check":
                    writer.write((verdict(message.line, message.term) + "\n").encode("utf-8"))
                    await writer.drain()
                    break  # one question, one answer
                self.lines.take(message)
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def keep(self, every: float = 2.0) -> None:
        """Open the socket while Live is on and close it while it is off."""
        while True:
            try:
                if self._enabled() and self._server is None:
                    await self.open()
                elif not self._enabled() and self._server is not None:
                    await self.close()
            except Exception:
                logger.warning("The Live shell socket could not be opened or closed", exc_info=True)
            await asyncio.sleep(every)

    def start(self) -> None:
        if sys.platform != "win32" and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self.keep(), name="heliox-live-shell")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.close()


def _uid() -> int:
    getuid = getattr(os, "getuid", None)
    return int(getuid()) if callable(getuid) else -1


# ── the integration files, shown before anything is written ──


ZSH_SCRIPT = """# Heliox Live: lets Heliox check the command you are typing (for example a risky rm -rf) before you press Enter.
# Only while Live is on; never a line that starts with a space. Remove: delete the line in ~/.zshrc that sources this
# file (Settings, Live, Terminal check, Remove does it), then this file.
zmodload zsh/net/socket 2>/dev/null || return 0
autoload -Uz add-zle-hook-widget 2>/dev/null || return 0
typeset -g _heliox_sock=__SOCKET__ _heliox_fd= _heliox_last=
_heliox_send() {
  setopt localoptions localtraps
  trap '' PIPE
  [[ -S $_heliox_sock ]] || { _heliox_fd=; return 0 }
  if [[ -z $_heliox_fd ]]; then
    zsocket $_heliox_sock 2>/dev/null || return 0
    _heliox_fd=$REPLY
  fi
  local line=${1//\\\\/\\\\\\\\}
  line=${line//$'\\n'/\\\\n}
  local term=${${TERM_PROGRAM:-${TERM:-terminal}}// /_}
  print -u $_heliox_fd -r -- "HLX1 line zsh $term $$ $line" 2>/dev/null || {
    exec {_heliox_fd}>&- 2>/dev/null
    _heliox_fd=
  }
}
_heliox_line_changed() {
  [[ $CONTEXT == start ]] || return 0
  [[ $BUFFER == $_heliox_last ]] && return 0
  _heliox_last=$BUFFER
  if [[ $BUFFER == [[:space:]]* ]]; then
    _heliox_send ""  # a line started with a space is never sent: Heliox forgets the last one
  else
    _heliox_send "$BUFFER"
  fi
}
add-zle-hook-widget line-pre-redraw _heliox_line_changed
"""

FISH_SCRIPT = """# Heliox Live: Enter asks Heliox first, so a risky command (for example rm -rf ~/) is not run by accident.
# Only while Live is on; never a line that starts with a space. Remove: delete this file (Settings, Live,
# Terminal check, Remove does it).
status is-interactive; or exit
set -g __heliox_sock __SOCKET__
function __heliox_enter
    set -l line (commandline)
    if test -S $__heliox_sock; and type -q nc; and not string match -qr '^\\s' -- "$line[1]"
        set -l text (string join '\\n' -- (string replace -a '\\\\' '\\\\\\\\' -- $line))
        set -l term (string replace -a ' ' _ -- (set -q TERM_PROGRAM; and echo $TERM_PROGRAM; or echo $TERM))
        set -l flags -U -w 1
        if nc -h 2>&1 | string match -q '*-N*'
            set flags $flags -N
        end
        set -l answer (printf 'HLX1 check fish %s %s %s\\n' $term $fish_pid "$text" | nc $flags $__heliox_sock 2>/dev/null)
        if string match -q 'warn*' -- "$answer"
            if test "$__heliox_warned" != "$line"
                set -g __heliox_warned "$line"
                set -l parts (string split \\t -- "$answer")
                echo
                echo "Heliox: $parts[2]. $parts[3] Press Enter again to run it anyway."
                commandline -f repaint
                return
            end
        end
    end
    set -g __heliox_warned ""
    commandline -f execute
end
bind \\r __heliox_enter
bind \\n __heliox_enter
"""


@dataclass(frozen=True)
class Plan:
    shell: str
    script_path: Path  # the file Heliox writes
    startup_file: Path | None  # the file one line is added to (zsh), or None (fish loads conf.d by itself)
    line: str  # the exact line added, or "" for fish
    remove: str  # how to take it out by hand


def plan(shell: str, home: Path, env: dict[str, str]) -> Plan:
    """What Add would do, shown before it is done. Pure."""
    data = Path(env.get("XDG_DATA_HOME") or home / ".local" / "share")
    config = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
    if shell == "zsh":
        script = data / "heliox" / "shell" / "heliox-live.zsh"
        startup = Path(env.get("ZDOTDIR") or home) / ".zshrc"
        line = f'[ -f "{script}" ] && source "{script}"  {MARKER} (delete this line to turn it off)'
        return Plan("zsh", script, startup, line, f"Delete the line ending in '{MARKER} ...' from {startup}.")
    if shell == "fish":
        script = config / "fish" / "conf.d" / "heliox-live.fish"
        return Plan("fish", script, None, "", f"Delete {script}.")
    raise ValueError("Only zsh and fish are supported; bash needs ble.sh, which is not supported yet.")


def script_for(shell: str, socket: Path) -> str:
    text = ZSH_SCRIPT if shell == "zsh" else FISH_SCRIPT
    quoted = "'" + socket.as_posix().replace("'", "'\\''") + "'"
    return text.replace("__SOCKET__", quoted)


def install(shell: str, home: Path, env: dict[str, str], socket: Path) -> Plan:
    """Write the script and, for zsh, add the one line. Called only after the person pressed Add."""
    chosen = plan(shell, home, env)
    chosen.script_path.parent.mkdir(parents=True, exist_ok=True)
    chosen.script_path.write_text(script_for(shell, socket), encoding="utf-8")
    if chosen.startup_file is not None:
        existing = chosen.startup_file.read_text(encoding="utf-8") if chosen.startup_file.exists() else ""
        if MARKER not in existing:
            sep = "" if not existing or existing.endswith("\n") else "\n"
            with chosen.startup_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{sep}{chosen.line}\n")
    return chosen


def uninstall(shell: str, home: Path, env: dict[str, str]) -> bool:
    """Remove exactly what install added. Returns whether anything was there."""
    chosen = plan(shell, home, env)
    removed = False
    if chosen.startup_file is not None and chosen.startup_file.exists():
        lines = chosen.startup_file.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if MARKER not in line]
        if len(kept) != len(lines):
            chosen.startup_file.write_text("".join(kept), encoding="utf-8")
            removed = True
    if chosen.script_path.exists():
        chosen.script_path.unlink()
        removed = True
    return removed


def installed(shell: str, home: Path, env: dict[str, str]) -> bool:
    chosen = plan(shell, home, env)
    if not chosen.script_path.exists():
        return False
    if chosen.startup_file is None:
        return True
    return chosen.startup_file.exists() and MARKER in chosen.startup_file.read_text(encoding="utf-8")


__all__ = [
    "FRESH_SECONDS",
    "Message",
    "Plan",
    "ShellLineServer",
    "ShellLines",
    "folder_problem",
    "install",
    "installed",
    "parse",
    "plan",
    "readable",
    "script_for",
    "socket_path",
    "uninstall",
    "verdict",
]
