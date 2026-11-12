"""A real run of Live's terminal check in zsh and fish, for the ``platform-readers`` CI job and the manual
``shell-check`` workflow (ubuntu and macOS).

Each shell runs interactively in a pseudo-terminal that is its controlling terminal (its own session, a 24x100
window), in an empty home folder, with the integration installed the way Settings' Add installs it
(``pilot.live.shell_line.install``) and Heliox's socket served by the real ``ShellLineServer`` in a short folder only
this user can open. Nothing is typed until the shell's own prompt (``HLXREADY``) is on screen. The command used is
``git push origin main --force`` in an empty folder: Live calls it risky, and if it ran anyway git only says the
folder is not a repository.

- zsh (started with ``-d``: no system-wide start-up files, so nothing like a compinit question eats the first keys):
  the line being typed reaches Heliox before Enter; a line started with a space never does (and clears the last
  one); what is typed inside ``vared`` never does (the command line that starts vared rightly does); with Live off
  nothing is taken.
- fish: Enter on the risky command prints Heliox's warning and does not run it; Enter again runs it; an ordinary
  command runs at once (its output, not just the echoed command) with no warning.
- the dry run Settings does before offering Add passes for both shells here, and fails for fish with no nc on PATH.
- a Heliox that is down or hung never holds the shell: fish's Enter against a socket that accepts and never answers
  runs the command within the bound (BOUND_SECONDS plus BOUND_SLACK), and at once with the socket refusing; zsh keeps
  taking keys when Heliox has stopped reading (thousands of long lines into a socket nobody reads).

Each check prints one ``CHECK`` line (``SKIP`` when a shell is not installed). A failed check is followed by
``DIAG`` lines: the terminal transcript around the step, the hook file as installed, the shell's version, whether
zsh's socket modules load, the nc found and its flags, how the shell has Enter bound, and Heliox's log of
connections and messages. The exit code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.live import shell_line  # noqa: E402
from pilot.live.shell_line import (  # noqa: E402
    BOUND_SECONDS,
    BOUND_SLACK,
    ShellLines,
    ShellLineServer,
    dry_run,
    install,
    plan,
    script_for,
)

RISKY = "git push origin main --force"
READY = "HLXREADY"
failures = 0
LOG: list[str] = []  # what Heliox's socket saw, for DIAG lines
_ANSI = re.compile(r"\x1b\[[0-9;?<>= ]*[A-Za-z~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]|\x1b[=>]")


def plain(text: str) -> str:
    return _ANSI.sub("", text).replace("\r", "")


def check(name: str, ok: bool, detail: str = "", diagnose=None) -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    if not ok and diagnose is not None:
        try:
            diagnose()
        except Exception as exc:  # the diagnosis must never hide the result
            print(f"DIAG (diagnosis failed: {type(exc).__name__}: {exc})", flush=True)
    return ok


def skip(name: str, why: str) -> None:
    print(f"CHECK {name}: SKIP - {why}", flush=True)


def diag(label: str, text: str) -> None:
    for line in str(text).splitlines() or [""]:
        print(f"DIAG {label}: {line}", flush=True)


def run_quiet(argv: list[str], env: dict[str, str] | None = None) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=10, env=env, stdin=subprocess.DEVNULL)
    except Exception as exc:
        return f"({type(exc).__name__}: {exc})"
    return f"exit {done.returncode}: {(done.stdout + done.stderr).strip()[:600]}"


def nc_facts() -> None:
    nc = shutil.which("nc")
    diag("nc", f"{nc} -> {os.path.realpath(nc) if nc else None}")
    if nc:
        usage = subprocess.run([nc, "-h"], capture_output=True, text=True, timeout=10)
        text = usage.stdout + usage.stderr
        diag("nc -h (first lines)", "\n".join(text.splitlines()[:3]))
        diag("nc flags the fish hook will use", "-U -w 1" + (" -N" if "-N" in text else " (no -N)"))


class Terminal:
    """An interactive shell on a pseudo-terminal that is its controlling terminal: type keys, read what it printed."""

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))

        def own_terminal() -> None:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)  # runs after setsid, with the pty already on stdin

        self.proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=cwd,
            close_fds=True,
            start_new_session=True,
            preexec_fn=own_terminal,  # noqa: PLW1509 - a test helper; no threads have started a lock yet
        )
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

    async def prompt(self, after: int = 0, seconds: float = 20.0) -> bool:
        return await wait_for(lambda: READY in plain(self.read()[after:]), seconds)

    def tail(self, start: int = 0, size: int = 900) -> str:
        return repr(self.read()[start:][-size:])

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


def watch(server: ShellLineServer) -> None:
    """Log what reaches Heliox's socket: connections, lines, Enter questions and their answers."""
    original_client = server._client
    original_take = server.lines.take
    original_verdict = shell_line.verdict

    async def client(reader, writer):
        LOG.append(f"{time.monotonic():.2f} connection")
        await original_client(reader, writer)
        LOG.append(f"{time.monotonic():.2f} connection closed")

    def take(message):
        LOG.append(f"{time.monotonic():.2f} line {message.shell} session {message.session}: {message.line[:60]!r}")
        original_take(message)

    def verdict(line, term):
        answer = original_verdict(line, term)
        LOG.append(f"{time.monotonic():.2f} enter question {line[:60]!r} -> {answer[:60]!r}")
        return answer

    server._client = client
    server.lines.take = take
    shell_line.verdict = verdict


def server_log(since: int = 0) -> None:
    diag("heliox socket log", "\n".join(LOG[since:][-25:]) or "(nothing reached the socket)")


def zsh_diagnose(term: Terminal, start: int, home: Path, env: dict[str, str], since: int):
    def show() -> None:
        zsh = shutil.which("zsh") or "zsh"
        diag("zsh --version", run_quiet([zsh, "--version"]))
        diag(
            "zsh modules",
            run_quiet(
                [
                    zsh,
                    "-f",
                    "-c",
                    "zmodload zsh/net/socket && print socket-ok; zmodload zsh/zselect && print zselect-ok",
                ]
            ),
        )
        chosen = plan("zsh", home, env)
        diag(".zshrc", chosen.startup_file.read_text(encoding="utf-8") if chosen.startup_file else "")
        diag(
            "hook file", chosen.script_path.read_text(encoding="utf-8") if chosen.script_path.exists() else "(missing)"
        )
        diag("transcript", term.tail(start))
        server_log(since)
        # From inside the running zsh: is the module loaded, the widget and the hook registered, the socket there?
        before = len(term.read())
        probe = (
            "\x15print -r -- HLXD zle=$(zmodload -e zsh/zle && print yes || print no)"
            " fn=${+functions[_heliox_line_changed]} widget=${widgets[_heliox_line_changed]-none}"
            " hook=${widgets[zle-line-pre-redraw]-none} sock=$([[ -S $_heliox_sock ]] && print yes || print no)"
            " fd=${_heliox_fd:-none} quiet=$_heliox_quiet_until; zstyle -L ':zle-hook:*'\r"
        )
        for char in probe:
            os.write(term.master, char.encode())
        time.sleep(1.5)
        diag("inside the zsh", plain(term.read()[before:])[-700:])
        server_log(since)

    return show


def fish_diagnose(term: Terminal, start: int, env: dict[str, str], since: int):
    def show() -> None:
        fish = shutil.which("fish") or "fish"
        diag("fish --version", run_quiet([fish, "--version"]))
        nc_facts()
        hook = Path(env["XDG_CONFIG_HOME"]) / "fish" / "conf.d" / "heliox-live.fish"
        diag("hook file", hook.read_text(encoding="utf-8") if hook.exists() else "(missing)")
        diag(
            "enter binding (interactive fish with this config)",
            run_quiet([fish, "-i", "-c", "functions -q __heliox_enter; and echo function-defined; bind \\r"], env),
        )
        diag("transcript", term.tail(start))
        server_log(since)

    return show


async def zsh_checks(server: ShellLineServer, live_on: list[bool], root: Path) -> None:
    zsh = shutil.which("zsh")
    if not zsh:
        skip("zsh", "zsh is not installed")
        return
    home = root / "zsh-home"
    home.mkdir()
    env = base_env(home)
    # A prompt to wait for, before the line install adds; -d skips the system-wide files (Ubuntu's runs compinit).
    (home / ".zshrc").write_text(f"PS1='{READY}%# '\n", encoding="utf-8")
    install("zsh", home, env, server.path)
    term = Terminal([zsh, "-d", "-i"], env, root / "work")
    latest = server.lines.latest
    try:
        if not check(
            "zsh: the shell shows its prompt",
            await term.prompt(),
            "",
            zsh_diagnose(term, 0, home, env, 0),
        ):
            return
        since, start = len(LOG), len(term.read())
        await term.type(RISKY)
        got = await wait_for(lambda: bool(latest()) and latest()[1].line == RISKY)
        check(
            "zsh: the line being typed reaches Heliox before Enter",
            got,
            repr(latest()[1].line if latest() else None),
            zsh_diagnose(term, start, home, env, since),
        )
        since, start = len(LOG), len(term.read())
        await term.type("\x15")  # Ctrl-U: kill the line
        await term.type(" " + RISKY)
        cleared = await wait_for(lambda: latest() is None)
        check(
            "zsh: a line started with a space is never sent, and clears the last",
            cleared and got,
            "" if got else "not shown: no line had reached Heliox before it",
            zsh_diagnose(term, start, home, env, since),
        )
        since, start = len(LOG), len(term.read())
        await term.type("\x15vared -c heliox_x\r")
        await term.settle(0.8)
        await term.type(RISKY)
        await term.settle(1.0)
        # The command line that starts vared is a command line and is sent; what is typed inside vared never is.
        inside = latest()
        check(
            "zsh: what is typed inside vared is never sent",
            inside is None or inside[1].line != RISKY,
            repr(inside),
            zsh_diagnose(term, start, home, env, since),
        )
        await term.type("\r\x15")
        await term.settle(0.5)
        since, start = len(LOG), len(term.read())
        live_on[0] = False
        server.lines.clear()
        await term.type("echo heliox-off")
        await term.settle(1.0)
        check(
            "zsh: with Live off nothing is taken",
            latest() is None,
            repr(latest()),
            zsh_diagnose(term, start, home, env, since),
        )
        live_on[0] = True
    finally:
        term.close()


def output_line(text: str, line: str) -> bool:
    """Whether ``line`` was printed as a line of its own (a command's output), not just echoed as typed."""
    return re.search(rf"(?m)^{re.escape(line)}\s*$", plain(text)) is not None


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
    # As Settings does: the dry run picks the nc and flags that work here, and Add writes those.
    ready = await dry_run("fish", server.path)
    diag("fish dry run", "\n".join(ready.tried) or ready.problem)
    if not check("fish: Settings' test offers Add here", ready.ok, f"{ready.problem} {ready.install}".strip()):
        for name in ("Enter on a risky command prints Heliox's warning", "Enter again runs it"):
            skip(f"fish: {name}", "Add is refused here, so there is no hook to check")
        return
    install("fish", home, env, server.path, (ready.nc, ready.nc_flags))
    config = home / ".config" / "fish" / "config.fish"
    config.write_text(f"set -g fish_greeting ''\nfunction fish_prompt\n    printf '{READY}> '\nend\n", encoding="utf-8")
    term = Terminal([fish, "-i"], env, root / "work")
    try:
        if not check("fish: the shell shows its prompt", await term.prompt(), "", fish_diagnose(term, 0, env, 0)):
            return
        since, start = len(LOG), len(term.read())
        await term.type(RISKY + "\r")
        warned = await wait_for(lambda: "Heliox: That can overwrite" in plain(term.read()[start:]), 6.0)
        new = term.read()[start:]
        check(
            "fish: Enter on a risky command prints Heliox's warning",
            warned,
            "",
            fish_diagnose(term, start, env, since),
        )
        check(
            "fish: the risky command did not run",
            warned and "not a git repository" not in new,
            "" if warned else "not shown: no warning came, so Enter did nothing at all",
            fish_diagnose(term, start, env, since),
        )
        since, start = len(LOG), len(term.read())
        await term.type("\r")
        ran = await wait_for(lambda: "not a git repository" in term.read()[start:], 6.0)
        check("fish: Enter again runs it", ran, "", fish_diagnose(term, start, env, since))
        await term.prompt(start)
        since, start = len(LOG), len(term.read())
        await term.type("echo heliox-plain\r")
        printed = await wait_for(lambda: output_line(term.read()[start:], "heliox-plain"), 6.0)
        new = term.read()[start:]
        check(
            "fish: an ordinary command runs at once, with no warning",
            printed and "Heliox:" not in plain(new),
            "",
            fish_diagnose(term, start, env, since),
        )
    finally:
        term.close()


async def silent_socket(path: Path) -> tuple[asyncio.AbstractServer, list[asyncio.StreamWriter]]:
    """A Heliox that accepts and never reads or answers."""
    held: list[asyncio.StreamWriter] = []

    async def client(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return await asyncio.start_unix_server(client, path=str(path)), held


async def dry_run_checks(root: Path) -> None:
    for shell in ("zsh", "fish"):
        if not shutil.which(shell) or (shell == "fish" and not shutil.which("nc")):
            skip(f"dry run: {shell}", f"{shell} or nc is not installed")
            continue
        ready = await dry_run(shell)
        detail = f"answered {ready.answered_ms} ms, silent {ready.silent_ms} ms; {ready.problem}"
        check(
            f"dry run: {shell} can reach Heliox here, so Add is offered",
            ready.ok,
            detail,
            lambda ready=ready: (nc_facts(), diag("tried", "\n".join(ready.tried) or "(nothing)")),
        )
    fish = shutil.which("fish")
    if not fish:
        return
    bare = root / "bare-bin"
    bare.mkdir()
    (bare / "fish").symlink_to(fish)
    old = os.environ["PATH"]
    os.environ["PATH"] = str(bare)
    try:
        ready = await dry_run("fish", which=lambda name: str(bare / name) if (bare / name).exists() else None)
    finally:
        os.environ["PATH"] = old
    check(
        "dry run: fish with no nc is refused and says to install it",
        not ready.ok and "nc" in ready.problem,
        ready.problem,
    )


async def fish_bound_checks(root: Path) -> None:
    fish = shutil.which("fish")
    if not fish or not shutil.which("nc"):
        skip("fish: Enter is bounded", "fish or nc is not installed")
        return
    home = root / "fish-bound-home"
    home.mkdir()
    env = base_env(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    sock = root / "hung" / "live-shell.sock"
    ready = await dry_run("fish")
    if not ready.ok:
        skip("fish: Enter is bounded", f"Add is refused here: {ready.problem}")
        return
    install("fish", home, env, sock, (ready.nc, ready.nc_flags))
    (home / ".config" / "fish" / "config.fish").write_text(
        f"set -g fish_greeting ''\nfunction fish_prompt\n    printf '{READY}> '\nend\n", encoding="utf-8"
    )
    server, held = await silent_socket(sock)
    term = Terminal([fish, "-i"], env, root / "work")
    limit = BOUND_SECONDS + BOUND_SLACK + 0.5  # and the typing of the Enter itself
    loop = asyncio.get_running_loop()
    try:
        if not check(
            "fish (bound): the shell shows its prompt", await term.prompt(), "", fish_diagnose(term, 0, env, 0)
        ):
            return
        start = len(term.read())
        await term.type("echo heliox-bounded", gap=0.01)
        began = loop.time()
        os.write(term.master, b"\r")
        ran = await wait_for(lambda: output_line(term.read()[start:], "heliox-bounded"), 10)
        waited = loop.time() - began
        check(
            "fish: with Heliox hung (accepts, never answers) Enter runs the command within the bound",
            ran and waited <= limit,
            f"{waited:.2f} s, bound {BOUND_SECONDS} s",
            fish_diagnose(term, start, env, len(LOG)),
        )
        server.close()
        for writer in held:
            writer.close()
        await server.wait_closed()
        sock.unlink(missing_ok=True)
        # A Heliox that died: its socket file is still there but nothing listens, so connecting is refused at once.
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(str(sock))
        await term.prompt(start)
        start = len(term.read())
        await term.type("echo heliox-down", gap=0.01)
        began = loop.time()
        os.write(term.master, b"\r")
        ran = await wait_for(lambda: output_line(term.read()[start:], "heliox-down"), 10)
        waited = loop.time() - began
        check(
            "fish: with Heliox not running Enter does not wait",
            ran and waited < BOUND_SECONDS,
            f"{waited:.2f} s",
            fish_diagnose(term, start, env, len(LOG)),
        )
        dead.close()
    finally:
        term.close()


async def zsh_hung_check(root: Path) -> None:
    zsh = shutil.which("zsh")
    if not zsh:
        skip("zsh: a hung Heliox", "zsh is not installed")
        return
    sock = root / "hung-zsh" / "live-shell.sock"
    server, held = await silent_socket(sock)
    script = root / "hook.zsh"
    script.write_text(script_for("zsh", sock), encoding="utf-8")
    body = 'source "$1"; line=${(l:400::x:)}; repeat 5000 _heliox_send "git status $line"; print sent'
    loop = asyncio.get_running_loop()
    began = loop.time()
    proc = await asyncio.create_subprocess_exec(
        zsh, "-f", "-c", body, "zsh", str(script), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 20)
        text = out.decode(errors="replace").strip()
        done = text.endswith("sent")
    except asyncio.TimeoutError:
        proc.kill()
        text, done = "(killed after 20 s)", False
    took = loop.time() - began
    check(
        "zsh: typing never waits on a Heliox that stopped reading (2 MB of lines, nobody reading)",
        done,
        f"{took:.2f} s",
        lambda: diag("zsh output", text[-600:]),
    )
    for writer in held:
        writer.close()
    server.close()
    await server.wait_closed()


async def main() -> int:
    # Under /tmp: a socket's path must stay under 104 bytes, and macOS's own temp folder is long.
    with tempfile.TemporaryDirectory(prefix="hlx-sh-", dir="/tmp") as folder:
        root = Path(folder)
        (root / "work").mkdir()
        runtime = root / "rt"
        live_on = [True]
        server = ShellLineServer(ShellLines(), enabled=lambda: live_on[0], path=runtime / "heliox" / "live-shell.sock")
        watch(server)  # before open: the socket keeps the handler it was opened with
        diag("versions", f"{run_quiet(['zsh', '--version'])} | {run_quiet(['fish', '--version'])}")
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
            await dry_run_checks(root)
            await fish_bound_checks(root)
            await zsh_hung_check(root)
            await server.close()
            check("the socket is gone when Heliox closes it", not server.path.exists())
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
