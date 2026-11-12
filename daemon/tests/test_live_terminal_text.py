"""kitty and WezTerm read through their own command-line control (pilot.live.terminal_text): the line the cursor is
on, after the prompt, with the shell hook's rules; asked in the background, given up on quickly."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.live import terminal_text as T
from pilot.live.detectors import TextContext, detect
from pilot.live.terminal_text import (
    KITTY_SETUP,
    TerminalText,
    command_part,
    kitty_addresses,
    kitty_cursor_line,
    kitty_focused_window,
    kitty_socket_setting,
    screen_row,
    typed_line,
    wezterm_cursor_row,
    wezterm_focused_pane,
    which_terminal,
)

SCREEN = "vk@box ~/src $ ls\nREADME.md  src\nvk@box ~/src $ git push origin main --force\n\n\n"


def test_which_terminal():
    assert which_terminal("wezterm-gui") == which_terminal("WezTerm") == "wezterm"
    assert which_terminal("kitty") == which_terminal("kitty.app") == "kitty"
    assert which_terminal("Terminal") == which_terminal("kittens-game") == which_terminal("python3") == ""


def test_only_what_follows_the_prompt_is_read_with_the_shell_hooks_rules():
    assert command_part("vk@box ~/src $ git push origin main --force") == "git push origin main --force"
    assert command_part("~/src ❯ rm -rf build") == "rm -rf build"
    assert command_part("PS C:\\> Remove-Item x") == "Remove-Item x"
    assert typed_line("\x1b[32mvk@box\x1b[0m $ git status") == "git status"
    assert typed_line("vk@box $  git push --force") is None  # a command started with a space is never read
    key = "export TOKEN=" + "A" * 40
    assert typed_line(f"$ {key}") == "export "  # nothing from the key-looking string onwards
    assert typed_line("   ") is None


def test_wezterm_answers():
    clients = json.dumps(
        [
            {"focused_pane_id": 3, "idle_time": {"secs": 40, "nanos": 0}},
            {"focused_pane_id": 7, "idle_time": {"secs": 0, "nanos": 5000}},
        ]
    )
    assert wezterm_focused_pane(clients) == 7  # the client the person is using
    assert wezterm_focused_pane("not json") is None and wezterm_focused_pane("[]") is None
    panes = json.dumps([{"pane_id": 7, "cursor_y": 2}, {"pane_id": 3, "cursor_y": 0}])
    assert wezterm_cursor_row(panes, 7) == 2 and wezterm_cursor_row(panes, 9) is None
    assert screen_row(SCREEN, 2) == "vk@box ~/src $ git push origin main --force"
    assert screen_row(SCREEN, None) == "vk@box ~/src $ git push origin main --force"  # last line with text


def test_kitty_answers_and_its_setting():
    assert kitty_socket_setting("allow_remote_control socket-only\nlisten_on unix:/tmp/kitty\n") == (
        True,
        "unix:/tmp/kitty",
    )
    assert kitty_socket_setting("# allow_remote_control yes\n") == (False, "")
    assert kitty_socket_setting("allow_remote_control password\nlisten_on unix:/tmp/k") == (False, "unix:/tmp/k")
    assert kitty_addresses("unix:/tmp/kitty", 42) == ["unix:/tmp/kitty-42", "unix:/tmp/kitty"]
    assert kitty_addresses("unix:/tmp/kitty-{kitty_pid}", 42) == ["unix:/tmp/kitty-42"]
    assert kitty_addresses("tcp:localhost:5000", 42) == []  # never over the network
    tree = json.dumps(
        [
            {"is_focused": False, "tabs": [{"is_focused": True, "windows": [{"id": 1, "is_focused": True}]}]},
            {
                "is_focused": True,
                "tabs": [
                    {"is_focused": False, "windows": [{"id": 2, "is_focused": True}]},
                    {"is_focused": True, "windows": [{"id": 3, "is_focused": False}, {"id": 4, "is_focused": True}]},
                ],
            },
        ]
    )
    assert kitty_focused_window(tree) == 4
    assert (
        kitty_cursor_line(SCREEN.rstrip("\n") + "\x1b[3;45H\x1b[2 q") == "vk@box ~/src $ git push origin main --force"
    )


class Fake:
    """Answers the terminal commands from a table; records what was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.asked: list[list[str]] = []

    async def __call__(self, argv, timeout):
        self.asked.append(argv)
        for needle, answer in self.answers.items():
            if needle in " ".join(argv):
                return answer
        return 1, ""


WEZTERM = {
    "list-clients": (0, json.dumps([{"focused_pane_id": 7, "idle_time": {"secs": 0, "nanos": 0}}])),
    "cli list --format": (0, json.dumps([{"pane_id": 7, "cursor_y": 2}])),
    "get-text --pane-id 7": (0, SCREEN),
}


@pytest.mark.asyncio
async def test_wezterm_is_read_and_kept_only_while_fresh():
    clock = [0.0]
    fake = Fake(WEZTERM)
    reader = TerminalText(runner=fake, which=lambda name: f"/usr/bin/{name}", env={}, clock=lambda: clock[0])
    seen = await reader.refresh("wezterm")
    assert seen.line == "git push origin main --force" and seen.pane == "wezterm:7"
    assert reader.latest("wezterm") is seen and reader.latest("kitty") is None
    clock[0] = T.FRESH_SECONDS + 0.1
    assert reader.latest("wezterm") is None


@pytest.mark.asyncio
async def test_a_terminal_that_does_not_answer_gives_nothing_and_says_why():
    async def hung(argv, timeout):
        return None, ""  # killed after CALL_SECONDS

    reader = TerminalText(runner=hung, which=lambda name: f"/usr/bin/{name}", env={})
    assert await reader.refresh("wezterm") is None
    assert reader.status()["wezterm"]["problem"] == "WezTerm did not say which pane has focus."
    assert T.CALL_SECONDS <= 1.0


@pytest.mark.asyncio
async def test_kitty_without_remote_control_is_never_asked_and_settings_says_what_to_add(tmp_path):
    fake = Fake({})
    reader = TerminalText(runner=fake, which=lambda name: f"/usr/bin/{name}", env={}, home=tmp_path)
    assert await reader.refresh("kitty", pid=42) is None
    assert fake.asked == []
    assert reader.status()["kitty"]["needs"] == KITTY_SETUP and "listen_on unix:/tmp/kitty" in KITTY_SETUP


@pytest.mark.asyncio
async def test_kitty_with_remote_control_is_read_from_the_kitty_in_front(tmp_path):
    conf = tmp_path / ".config" / "kitty" / "kitty.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text("allow_remote_control socket-only\nlisten_on unix:/tmp/kitty\n", encoding="utf-8")
    tree = json.dumps(
        [{"is_focused": True, "tabs": [{"is_focused": True, "windows": [{"id": 4, "is_focused": True}]}]}]
    )
    fake = Fake(
        {
            "--to unix:/tmp/kitty-42 ls": (0, tree),
            "get-text --match id:4": (0, SCREEN.rstrip("\n") + "\x1b[3;45H"),
        }
    )
    reader = TerminalText(runner=fake, which=lambda name: f"/usr/bin/{name}", env={}, home=tmp_path)
    seen = await reader.refresh("kitty", pid=42)
    assert seen.line == "git push origin main --force" and seen.pane == "kitty:unix:/tmp/kitty-42:4"
    assert all("--to" in argv and argv[argv.index("--to") + 1].startswith("unix:") for argv in fake.asked)


@pytest.mark.asyncio
async def test_poke_asks_in_the_background_and_not_more_often_than_live_looks():
    clock = [0.0]
    fake = Fake(WEZTERM)
    reader = TerminalText(runner=fake, which=lambda name: f"/usr/bin/{name}", env={}, clock=lambda: clock[0])
    reader.poke("wezterm")
    reader.poke("wezterm")  # a moment later: not asked again
    await asyncio.sleep(0)
    await reader._task
    assert len([a for a in fake.asked if "list-clients" in a]) == 1
    assert reader.latest("wezterm").line == "git push origin main --force"
    reader.poke("Terminal")  # not one of these terminals: nothing
    assert len(fake.asked) == 3


@pytest.mark.asyncio
async def test_live_sees_the_wezterm_line_as_a_terminal_line_and_catches_a_risky_command():
    from pilot.personal.wiring import shell_observation

    clock = [0.0]
    reader = TerminalText(runner=Fake(WEZTERM), which=lambda name: f"/usr/bin/{name}", env={}, clock=lambda: clock[0])
    await reader.refresh("wezterm")
    reader._asked_at = 0.0  # asked just now
    server = SimpleNamespace(
        config=SimpleNamespace(live=SimpleNamespace(read_focused_text=True)), _terminal_text=reader
    )
    obs = shell_observation(server, "wezterm-gui", "~/src")
    assert obs.text == "git push origin main --force" and obs.field == "wezterm:7"
    ctx = TextContext(today=date(2026, 11, 12), app=obs.app, title=obs.title, source="focused_field")
    assert [c.kind for c in detect(obs.text, ctx, kinds=("risky_command",))] == ["risky_command"]
    server.config.live.read_focused_text = False  # reading fields turned off: not asked, not read
    assert shell_observation(server, "wezterm-gui", "~/src") is None
    assert shell_observation(SimpleNamespace(config=server.config), "Notepad", "x") is None


def test_no_path_of_this_computer_in_the_reader():
    source = Path(T.__file__).read_text(encoding="utf-8")
    assert "marcu" not in source and "C:\\\\Users" not in source
