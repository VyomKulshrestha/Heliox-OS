"""Live's terminal check for zsh and fish: the decisions everywhere, the socket on macOS and Linux."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from pilot.live import shell_line
from pilot.live.shell_line import (
    Message,
    ShellLines,
    ShellLineServer,
    folder_problem,
    install,
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
        "string match -qr '^\\s'" in fish and "nc $flags $__heliox_sock" in fish and "bind \\r __heliox_enter" in fish
    )


@POSIX
@pytest.mark.asyncio
async def test_the_socket_takes_lines_answers_enter_and_exists_only_while_live_is_on(tmp_path):
    folder = tmp_path / "rt"
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
    on[0] = False  # Live off: what arrives is ignored
    reader, writer = await asyncio.open_unix_connection(str(server.path))
    writer.write(b"HLX1 check fish kitty 9 git push origin main --force\n")
    await writer.drain()
    writer.write_eof()
    assert await asyncio.wait_for(reader.read(), 5) == b""
    writer.close()
    await server.close()
    assert not server.path.exists()


@POSIX
@pytest.mark.asyncio
async def test_a_folder_others_can_open_is_refused(tmp_path):
    folder = tmp_path / "open"
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
