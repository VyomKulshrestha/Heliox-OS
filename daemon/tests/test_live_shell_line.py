"""Live's terminal check for zsh and fish: the decisions everywhere, the socket on macOS and Linux."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

from pilot.live import shell_line
from pilot.live.shell_line import (
    BOUND_SECONDS,
    Message,
    Ready,
    ShellLines,
    ShellLineServer,
    dry_run,
    folder_problem,
    install,
    install_hint,
    installed,
    parse,
    plan,
    readable,
    script_for,
    socket_path,
    uninstall,
    verdict,
)

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets; Windows terminals are read directly")


@pytest.fixture
def short_folder():
    """A folder with a short path for a Unix socket: pytest's tmp_path on macOS is longer than the 104 bytes a
    socket path may have ("AF_UNIX path too long" on the macOS runners, 12 November)."""
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="hlx-", dir="/tmp"))
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def test_the_socket_lives_in_a_folder_only_this_user_can_open():
    assert socket_path({"XDG_RUNTIME_DIR": "/run/user/1000"}, 1000) == Path("/run/user/1000/heliox/live-shell.sock")
    assert socket_path({"TMPDIR": "/var/folders/ab/T/"}, 501) == Path("/var/folders/ab/T/heliox-501/live-shell.sock")
    assert socket_path({}, 1000) == Path("/tmp/heliox-1000/live-shell.sock")
    assert folder_problem(0o700, 1000, 1000) == ""
    assert folder_problem(0o755, 1000, 1000) == "other users can open the folder"
    assert folder_problem(0o700, 0, 1000) == "the folder belongs to another user"


def test_messages_and_what_may_be_read():
    message = parse("HLX1 line zsh kitty 4242 git push --force \\\\ x\\nnext\n")
    assert message == Message("line", "zsh", "kitty", "4242", "git push --force \\ x\nnext")
    assert parse("HLX1 line bash kitty 1 ls") is None  # only the shells that are supported
    assert parse("hello") is None and parse("HLX1 run zsh t 1 ls") is None
    assert readable(" rm -rf ~/") is None  # a line started with a space is never read
    assert readable("") is None
    assert readable("export TOKEN=sk-proj-abcdefghijklmnopqrstuvwx && rm -rf /") == "export "
    assert readable("curl -H 'auth: a8F3kQ9zLmP2xR7vT1wY6bN4cJ0dE5gH' https://x") == "curl -H 'auth: "


def test_enter_in_fish_is_answered_from_lives_own_risky_command_check():
    assert verdict("git push origin main --force", "kitty").startswith("warn\tThat can overwrite teammates' commits\t")
    assert verdict("ls -la", "kitty") == "ok"
    assert verdict(" git push origin main --force", "kitty") == "ok"  # a leading space: not read, not checked


def test_lines_are_kept_only_while_fresh():
    clock = [0.0]
    lines = ShellLines(clock=lambda: clock[0])
    lines.take(Message("line", "zsh", "kitty", "1", "rm -rf ~/"))
    key, seen = lines.latest()
    assert key == "zsh:1" and seen.line == "rm -rf ~/"
    lines.take(Message("line", "zsh", "kitty", "1", " secret stuff"))  # a space-led line clears the session
    assert lines.latest() is None
    lines.take(Message("line", "zsh", "kitty", "2", "ls"))
    clock[0] = shell_line.FRESH_SECONDS + 1
    assert lines.latest() is None


def test_add_shows_exactly_what_it_will_do_and_remove_takes_exactly_that_out(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").write_text("export EDITOR=vim", encoding="utf-8")
    env: dict[str, str] = {}
    shown = plan("zsh", home, env)
    assert shown.startup_file == home / ".zshrc" and shown.script_path.name == "heliox-live.zsh"
    assert shown.line.startswith(f'[ -f "{shown.script_path}" ] && source "{shown.script_path}"')
    assert not installed("zsh", home, env)
    install("zsh", home, env, Path("/tmp/heliox-1000/live-shell.sock"))
    install("zsh", home, env, Path("/tmp/heliox-1000/live-shell.sock"))  # twice adds it once
    text = (home / ".zshrc").read_text(encoding="utf-8")
    assert text == f"export EDITOR=vim\n{shown.line}\n" and installed("zsh", home, env)
    assert "/tmp/heliox-1000/live-shell.sock" in shown.script_path.read_text(encoding="utf-8")
    assert uninstall("zsh", home, env)
    assert (home / ".zshrc").read_text(encoding="utf-8") == "export EDITOR=vim\n"
    assert not shown.script_path.exists() and not installed("zsh", home, env)

    fish = plan("fish", home, {"XDG_CONFIG_HOME": str(tmp_path / "cfg")})
    assert fish.startup_file is None and fish.script_path == tmp_path / "cfg" / "fish" / "conf.d" / "heliox-live.fish"
    install("fish", home, {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}, Path("/tmp/s.sock"))
    assert fish.script_path.exists()
    assert uninstall("fish", home, {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}) and not fish.script_path.exists()
    with pytest.raises(ValueError, match="ble.sh"):
        plan("bash", home, env)


def test_the_scripts_keep_the_rules():
    zsh = script_for("zsh", Path("/tmp/it's/live-shell.sock"))
    assert "_heliox_sock='/tmp/it'\\''s/live-shell.sock'" in zsh  # quoted for the shell
    assert "[[ $CONTEXT == start ]] || return 0" in zsh  # not vared, not a program's prompt
    assert "[[ $BUFFER == [[:space:]]* ]]" in zsh and "trap '' PIPE" in zsh
    fish = script_for("fish", Path("/tmp/s.sock"))
    assert (
        "string match -qr '^\\s'" in fish
        and "nc $__heliox_nc $__heliox_sock" in fish
        and "bind \\r __heliox_enter" in fish
    )
    # Enter's wait is bounded, and nc's flags are worked out once when fish starts, not on every Enter.
    assert "set -g __heliox_nc -U -w 1" in fish and fish.index("nc -h") < fish.index("function __heliox_enter")
    assert "type -q nc; or exit" in fish
    # zsh never waits on a Heliox that stopped reading, and leaves it alone for a while after.
    assert "zselect -t 0 -w $_heliox_fd" in zsh and "_heliox_quiet_until" in zsh
    assert "[[ -o interactive ]] && add-zle-hook-widget" in zsh


@POSIX
@pytest.mark.asyncio
async def test_the_socket_takes_lines_answers_enter_and_exists_only_while_live_is_on(short_folder):
    folder = short_folder / "rt"
    on = [True]
    server = ShellLineServer(ShellLines(), enabled=lambda: on[0], path=folder / "live-shell.sock", uid=os.getuid())
    assert await server.open()
    assert oct(folder.stat().st_mode & 0o777) == "0o700" and oct(server.path.stat().st_mode & 0o777) == "0o600"
    reader, writer = await asyncio.open_unix_connection(str(server.path))
    writer.write(b"HLX1 line zsh kitty 7 git push origin main --force\n")
    await writer.drain()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if server.lines.latest():
            break
    assert server.lines.latest()[1].line == "git push origin main --force"
    writer.close()
    reader, writer = await asyncio.open_unix_connection(str(server.path))
    writer.write(b"HLX1 check fish kitty 8 git push origin main --force\n")
    await writer.drain()
    answer = await asyncio.wait_for(reader.readline(), 5)
    assert answer.decode().startswith("warn\t")
    writer.close()
    on[0] = False  # Live off: lines are ignored, and Enter is answered "ok" at once rather than kept waiting
    server.lines.clear()
    reader, writer = await asyncio.open_unix_connection(str(server.path))
    writer.write(b"HLX1 line zsh kitty 7 git push origin main --force\n")
    writer.write(b"HLX1 check fish kitty 9 git push origin main --force\n")
    await writer.drain()
    assert await asyncio.wait_for(reader.readline(), 5) == b"ok\n"
    assert server.lines.latest() is None
    writer.close()
    await server.close()
    assert not server.path.exists()


@POSIX
@pytest.mark.asyncio
async def test_a_folder_others_can_open_is_refused(short_folder):
    folder = short_folder / "open"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    server = ShellLineServer(ShellLines(), enabled=lambda: True, path=folder / "s.sock", uid=os.getuid())
    assert not await server.open() and server.problem == "other users can open the folder"
    assert not (folder / "s.sock").exists()


def test_a_fresh_zsh_line_is_what_live_sees_and_its_risky_command_is_caught():
    from datetime import date
    from types import SimpleNamespace

    from pilot.live.detectors import TextContext, detect
    from pilot.personal.wiring import shell_observation

    lines = ShellLines()
    lines.take(Message("line", "zsh", "xterm-kitty", "1", "git push origin main --force"))
    server = SimpleNamespace(_shell_lines_server=SimpleNamespace(listening=True, lines=lines))
    seen = shell_observation(server, "kitty", "~/repo")
    assert seen.title == "~/repo (zsh)" and seen.source == "focused_field" and seen.text.endswith("--force")
    ctx = TextContext(today=date(2026, 11, 9), app=seen.app, title=seen.title)
    assert [c.kind for c in detect(seen.text, ctx)] == ["risky_command"]
    assert shell_observation(SimpleNamespace(), "kitty", "x") is None


# ── the dry run before Add ──


def test_what_to_install_is_said_for_the_kind_of_computer():
    nothing = lambda _name: None  # noqa: E731
    mac = install_hint("nc", "darwin", nothing)
    assert mac.startswith("Nothing needs installing: macOS's own nc (/usr/bin/nc)") and "brew uninstall netcat" in mac
    assert "brew install netcat" not in mac  # Homebrew's netcat is GNU netcat: it has no -U
    zypper = lambda name: "/usr/bin/zypper" if name == "zypper" else None  # noqa: E731
    assert install_hint("nc", "linux", zypper) == "Install the OpenBSD netcat (often packaged as netcat-openbsd)."
    apt = lambda name: "/usr/bin/apt-get" if name == "apt-get" else None  # noqa: E731
    assert install_hint("nc", "linux", apt) == "Install the OpenBSD netcat: sudo apt install netcat-openbsd."
    pacman = lambda name: "/usr/bin/pacman" if name == "pacman" else None  # noqa: E731
    assert "openbsd-netcat" in install_hint("nc", "linux", pacman)
    assert "netcat-openbsd" in install_hint("nc", "linux", nothing)
    assert "brew install zsh" in install_hint("zsh-modules", "darwin", nothing)
    assert "zsh/net/socket" in install_hint("zsh-modules", "linux", nothing)


@pytest.mark.asyncio
async def test_no_shell_no_add():
    missing = await dry_run("fish", which=lambda _name: None, platform="linux")
    assert not missing.ok and missing.problem == "fish is not installed here."
    assert not (await dry_run("bash", platform="linux")).ok
    assert missing.as_dict()["bound_ms"] == int(BOUND_SECONDS * 1000) == 1000


@pytest.mark.asyncio
async def test_install_is_refused_when_the_dry_run_fails(tmp_path, monkeypatch):
    from pilot.personal import wiring

    async def failing(shell, socket=None, **_):
        return Ready(shell, False, "fish needs nc (netcat) to reach Heliox, and there is none here.", "Install nc.")

    monkeypatch.setattr(shell_line, "dry_run", failing)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    server = type("S", (), {})()
    server._shell_lines_server = type("F", (), {"listening": False, "path": tmp_path / "s.sock", "problem": ""})()
    answer = await wiring.live_shell_rpc(server, {"action": "install", "shell": "fish"})
    assert answer["status"] == "error" and answer["message"].startswith("Not added: fish needs nc")
    assert answer["ready"]["fish"]["install"] == "Install nc."
    assert not (tmp_path / "cfg" / "fish" / "conf.d" / "heliox-live.fish").exists()  # nothing written


def _stub(folder: Path, body: str) -> str:
    path = folder / "fakeshell"
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o700)
    return str(path)


@POSIX
@pytest.mark.asyncio
async def test_an_nc_that_never_gives_up_is_caught_and_add_is_not_offered(tmp_path):
    """The silent socket accepts and never answers: an Enter that would wait on it past the bound fails the dry run."""
    fake = _stub(tmp_path, 'case "$*" in *silent.sock*) sleep 30;; *) echo ok;; esac')
    ready = await dry_run("fish", which=lambda name: fake if name == "fish" else None, platform="linux", timeout=0.5)
    assert not ready.ok and "does not give up" in ready.problem
    assert ready.silent_ms is not None and ready.silent_ms >= int(BOUND_SECONDS * 1000)


@POSIX
@pytest.mark.asyncio
async def test_a_shell_that_cannot_reach_the_socket_is_caught(tmp_path):
    fake = _stub(tmp_path, "exit 3")
    zsh = await dry_run("zsh", which=lambda name: fake if name == "zsh" else None, platform="linux")
    assert not zsh.ok and "zsh/net/socket" in zsh.problem
    fish = await dry_run("fish", which=lambda name: fake if name == "fish" else None, platform="darwin")
    assert not fish.ok and fish.install.startswith("Nothing needs installing: macOS's own nc")
    quiet = _stub(tmp_path, "echo nothing")
    nc = await dry_run("fish", which=lambda name: quiet if name == "fish" else None, platform="linux")
    assert not nc.ok and "-U" in nc.problem


@POSIX
@pytest.mark.asyncio
@pytest.mark.skipif(not (shutil.which("fish") and shutil.which("nc")), reason="needs fish and nc")
async def test_real_fish_reaches_heliox_and_gives_up_on_a_silent_one_within_the_bound():
    ready = await dry_run("fish")
    assert ready.ok, ready.problem
    assert ready.silent_ms <= int((BOUND_SECONDS + shell_line.BOUND_SLACK) * 1000)


@POSIX
@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("zsh"), reason="needs zsh")
async def test_real_zsh_reaches_heliox():
    ready = await dry_run("zsh")
    assert ready.ok, ready.problem


# ── a socket path too long for the system (a long TMPDIR on macOS) ──


def test_a_socket_path_too_long_falls_back_to_tmp_and_says_so_when_even_that_is():
    from pilot.live.shell_line import MAX_SOCKET_PATH, fits

    long = "/var/folders/" + "x" * 90 + "/T"
    assert socket_path({"TMPDIR": long}, 501) == Path("/tmp/heliox-501/live-shell.sock")
    assert socket_path({"XDG_RUNTIME_DIR": "/run/user/" + "9" * 100, "TMPDIR": "/tmp"}, 7) == Path(
        "/tmp/heliox-7/live-shell.sock"
    )
    assert socket_path({"TMPDIR": "/var/folders/ab/T"}, 501) == Path("/var/folders/ab/T/heliox-501/live-shell.sock")
    assert fits(Path("/" + "a" * (MAX_SOCKET_PATH - 1))) and not fits(Path("/" + "a" * MAX_SOCKET_PATH))


@pytest.mark.asyncio
async def test_open_refuses_a_path_too_long_and_says_why(tmp_path):
    server = ShellLineServer(ShellLines(), enabled=lambda: True, path=tmp_path / ("d" * 120) / "s.sock", uid=0)
    if sys.platform == "win32":
        assert not await server.open()  # nothing to open on Windows
        return
    assert not await server.open()
    assert server.problem.startswith("the socket's path is too long for this system")


def test_the_dry_runs_throwaway_folder_is_short_enough(monkeypatch, tmp_path):
    import tempfile

    from pilot.live.shell_line import fits, short_temp_folder

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / ("t" * 110)))
    made: list[str] = []
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix, dir: made.append(dir) or f"{dir}/{prefix}abc")
    folder = short_temp_folder("heliox-dry-")
    assert [Path(d) for d in made] == [Path("/tmp")] and fits(folder / "silent.sock")
