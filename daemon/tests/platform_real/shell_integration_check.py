"""A real run of Live's terminal check in zsh and fish, for the ``platform-readers`` CI job (ubuntu and macOS).

Each shell runs interactively in a pseudo-terminal, in an empty home folder, with the integration installed the way
Settings' Add installs it (``pilot.live.shell_line.install``) and Heliox's socket served by the real
``ShellLineServer`` in a folder only this user can open. Keys are typed one at a time. The command used is
``git push origin main --force`` in an empty folder: Live calls it risky, and if it ran anyway git only says the
folder is not a repository.

- zsh: the line being typed reaches Heliox before Enter; a line started with a space never does (and clears the last
  one); a line typed in ``vared`` never does; with Live off nothing is taken.
- fish: Enter on the risky command prints Heliox's warning and does not run it; Enter again runs it; an ordinary
  command runs at once with no warning.

Each check prints one ``CHECK`` line (``SKIP`` when a shell is not installed); the exit code is the number of failed
checks.
"""

from __future__ import annotations

import asyncio
import os
import pty
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.live.shell_line import ShellLines, ShellLineServer, install  # noqa: E402

RISKY = "git push origin main --force"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def skip(name: str, why: str) -> None:
    print(f"CHECK {name}: SKIP - {why}", flush=True)


class Terminal:
    """An interactive shell on a pseudo-terminal: type keys, read what it printed."""

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd, close_fds=True)
        os.close(slave)
        os.set_blocking(self.master, False)
        self.output = ""

    def read(self) -> str:
        while True:
            try:
                chunk = os.read(self.master, 65536)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            self.output += chunk.decode("utf-8", errors="replace")
        return self.output

    async def type(self, text: str, gap: float = 0.03) -> None:
        for char in text:
            os.write(self.master, char.encode())
            await asyncio.sleep(gap)
            self.read()

    async def settle(self, seconds: float) -> str:
        end = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < end:
            self.read()
            await asyncio.sleep(0.05)
        return self.read()

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=10)
        os.close(self.master)


async def wait_for(condition, seconds: float = 5.0) -> bool:
    end = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < end:
        if condition():
            return True
        await asyncio.sleep(0.05)
    return bool(condition())


def base_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ZDOTDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME")}
    env.update({"HOME": str(home), "TERM": "xterm-256color", "TERM_PROGRAM": "heliox-ci", "LANG": "C.UTF-8"})
    return env


async def zsh_checks(server: ShellLineServer, live_on: list[bool], root: Path) -> None:
    zsh = shutil.which("zsh")
    if not zsh:
        skip("zsh", "zsh is not installed")
        return
    home = root / "zsh-home"
    home.mkdir()
    env = base_env(home)
    install("zsh", home, env, server.path)
    term = Terminal([zsh, "-i"], env, root / "work")
    try:
        await term.settle(2.0)
        latest = server.lines.latest
        await term.type(RISKY)
        got = await wait_for(lambda: bool(latest()) and latest()[1].line == RISKY)
        check(
            "zsh: the line being typed reaches Heliox before Enter", got, repr(latest()[1].line if latest() else None)
        )
        await term.type("\x15")  # Ctrl-U: kill the line
        await term.type(" " + RISKY)
        cleared = await wait_for(lambda: latest() is None)
        check("zsh: a line started with a space is never sent, and clears the last", cleared)
        await term.type("\x15vared -c heliox_x\r")
        await term.settle(0.5)
        await term.type(RISKY)
        await term.settle(1.0)
        check("zsh: a line typed in vared is never sent", latest() is None, repr(latest()))
        await term.type("\r\x15")
        await term.settle(0.5)
        live_on[0] = False
        await term.type("echo heliox-off")
        await term.settle(1.0)
        check("zsh: with Live off nothing is taken", latest() is None, repr(latest()))
        live_on[0] = True
    finally:
        term.close()


async def fish_checks(server: ShellLineServer, root: Path) -> None:
    fish = shutil.which("fish")
    if not fish:
        skip("fish", "fish is not installed")
        return
    if not shutil.which("nc"):
        skip("fish", "nc is not installed")
        return
    home = root / "fish-home"
    home.mkdir()
    env = base_env(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    install("fish", home, env, server.path)
    term = Terminal([fish, "-i"], env, root / "work")
    try:
        await term.settle(3.0)
        start = len(term.read())
        await term.type(RISKY + "\r")
        out = await term.settle(3.0)
        new = out[start:]
        warned = "Heliox: That can overwrite" in new
        check("fish: Enter on a risky command prints Heliox's warning", warned, repr(new[-300:]))
        check("fish: the risky command did not run", "not a git repository" not in new, repr(new[-200:]))
        start = len(out)
        await term.type("\r")
        out = await term.settle(3.0)
        check("fish: Enter again runs it", "not a git repository" in out[start:], repr(out[start:][-200:]))
        start = len(out)
        await term.type("echo heliox-plain\r")
        out = await term.settle(3.0)
        new = out[start:]
        check(
            "fish: an ordinary command runs at once", "heliox-plain" in new and "Heliox:" not in new, repr(new[-200:])
        )
    finally:
        term.close()


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heliox-shell-") as folder:
        root = Path(folder)
        (root / "work").mkdir()
        runtime = root / "rt"
        live_on = [True]
        server = ShellLineServer(ShellLines(), enabled=lambda: live_on[0], path=runtime / "heliox" / "live-shell.sock")
        opened = await server.open()
        check("the socket opens in a folder only this user can open", opened, server.problem)
        if opened:
            mode = oct((runtime / "heliox").stat().st_mode & 0o777)
            check(
                "the socket folder is 0700 and the socket 0600",
                mode == "0o700" and oct(server.path.stat().st_mode & 0o777) == "0o600",
                mode,
            )
            await zsh_checks(server, live_on, root)
            await fish_checks(server, root)
            await server.close()
            check("the socket is gone when Heliox closes it", not server.path.exists())
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
