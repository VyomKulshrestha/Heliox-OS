"""What the zsh hook never sends: answers typed at prompts, in a real zsh, for the shell-check workflow and the
``platform-readers`` job (ubuntu and macOS). Reuses ``shell_integration_check``'s terminal (the pty as the shell's
controlling terminal, a prompt to wait for) and its log of everything that reaches Heliox's socket.

- ``read -s`` (zsh's own silent prompt): what is typed there is never sent.
- A program's own password prompt (Python's ``getpass``, which reads the terminal with echo off, the way sudo does;
  sudo itself asks the runner for no password): what is typed there is never sent.
- ``zmodload zsh/net/socket`` and ``zsh/zselect`` load in the zsh on PATH and, on macOS, in ``/bin/zsh``.

Each step prints one ``CHECK`` line (``SKIP`` with the reason); a failed one is followed by ``DIAG`` lines. The exit
code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import shell_integration_check as S  # noqa: E402

from pilot.live.shell_line import ShellLines, ShellLineServer, install  # noqa: E402

SECRET = "hunter2-heliox-ci"
failures = 0


def check(name: str, ok: bool, detail: str = "", diagnose=None) -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    if not ok and diagnose is not None:
        diagnose()
    return ok


def modules() -> None:
    shells = [shutil.which("zsh") or ""]
    if sys.platform == "darwin" and Path("/bin/zsh").exists():
        shells.append("/bin/zsh")
    for zsh in dict.fromkeys(s for s in shells if s):
        done = subprocess.run(
            [zsh, "-f", "-c", "zmodload zsh/net/socket && zmodload zsh/zselect && print loaded"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = subprocess.run([zsh, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
        check(
            f"zsh modules load in {zsh}",
            done.stdout.strip() == "loaded",
            f"{version}; {(done.stdout + done.stderr).strip()[:200]}",
        )


async def prompts(root: Path) -> None:
    zsh = shutil.which("zsh")
    if not zsh:
        print("CHECK zsh prompts: SKIP - zsh is not installed", flush=True)
        return
    server = ShellLineServer(ShellLines(), enabled=lambda: True, path=root / "rt" / "heliox" / "live-shell.sock")
    S.watch(server)
    if not check("the socket opens", await server.open(), server.problem):
        return
    home = root / "home"
    home.mkdir()
    env = S.base_env(home)
    (home / ".zshrc").write_text(f"PS1='{S.READY}%# '\n", encoding="utf-8")
    install("zsh", home, env, server.path)
    term = S.Terminal([zsh, "-d", "-i"], env, root)
    try:
        if not check("zsh shows its prompt", await term.prompt(), "", lambda: S.diag("transcript", term.tail())):
            return
        for name, command in (
            ("read -s", "read -s heliox_answer"),
            ("a program's own password prompt", f"{sys.executable} -c 'import getpass; getpass.getpass()'"),
        ):
            since, start = len(S.LOG), len(term.read())
            try:
                await term.type(command + "\r", gap=0.01)
                await term.settle(1.5)
                await term.type(SECRET + "\r")
                back = await term.prompt(start + len(command), 10.0)
            except Exception as exc:  # one step that raises is one failed CHECK; the next still runs
                check(f"zsh: what is typed at {name} is never sent", False, f"{type(exc).__name__}: {exc}")
                continue
            leaked = [line for line in S.LOG[since:] if SECRET in line]
            check(
                f"zsh: what is typed at {name} is never sent",
                back and not leaked,
                "" if back else "the prompt did not come back",
                lambda since=since, start=start: (S.diag("transcript", term.tail(start)), S.server_log(since)),
            )
    finally:
        term.close()
        await server.close()


def main() -> int:
    if not (sys.platform == "darwin" or sys.platform.startswith("linux")):
        print("CHECK zsh prompts: SKIP - macOS and Linux only", flush=True)
        return 0
    # Each part reports on its own: one that raises is one failed CHECK with the cause, never the end of the rest.
    try:
        if shutil.which("zsh"):
            modules()
    except Exception as exc:
        check("zsh modules: ran to its end", False, f"{type(exc).__name__}: {exc}")
    with tempfile.TemporaryDirectory(prefix="hlx-pr-", dir="/tmp") as folder:
        try:
            asyncio.run(prompts(Path(folder)))
        except Exception as exc:
            check("zsh prompts: ran to its end", False, f"{type(exc).__name__}: {exc}")
            S.diag("zsh prompts traceback", traceback.format_exc()[-1500:])
    return failures


if __name__ == "__main__":
    sys.exit(main())
