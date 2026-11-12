"""Values reach PowerShell and AppleScript as data: whole, and never in the code text."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys

import pytest

from pilot.system import script_values
from pilot.system.script_values import ENV_VALUE_LIMIT, osascript_command, powershell_values

# Each kind of quote PowerShell or AppleScript ends a string at, and the characters PowerShell, cmd and sh treat
# as code: ASCII ' and ", U+2018 to U+201B, a backtick, $ and a subexpression, a newline, ;, &, % and a pipe.
HOSTILE = 'it\'s ‘a’ ‚b‛ "q" `n $env:PATH $(Get-Date) \n next; & calc %PATH% | x'
ECHO = "[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes([string]$HelioxArgs['v']))"


def test_a_value_is_in_the_environment_and_never_in_the_script():
    with powershell_values("Write-Output $HelioxArgs['v']", {"v": HOSTILE}) as (args, env):
        assert env["HELIOX_ARG_V"] == HOSTILE
        assert all(HOSTILE not in arg for arg in args)
        for part in ("it's", "‘a", "$(Get-Date)", "& calc", "%PATH%"):
            assert part not in args[-1]


def test_a_long_value_goes_through_an_owner_only_file_removed_afterwards():
    long_text = HOSTILE * (ENV_VALUE_LIMIT // len(HOSTILE) + 2)
    with powershell_values("Write-Output $HelioxArgs['v']", {"v": long_text}) as (args, env):
        path = env["HELIOX_ARGFILE_V"]
        assert "HELIOX_ARG_V" not in env and long_text not in args[-1]
        with open(path, encoding="utf-8", newline="") as handle:
            assert handle.read() == long_text
        if os.name != "nt":
            assert os.stat(path).st_mode & 0o077 == 0, "readable by its owner only"
    assert not os.path.exists(path)


def test_the_file_is_removed_when_the_command_fails_half_way():
    long_text = "x" * (ENV_VALUE_LIMIT + 1)
    with pytest.raises(RuntimeError), powershell_values("Write-Output 1", {"v": long_text}) as (_args, env):
        path = env["HELIOX_ARGFILE_V"]
        raise RuntimeError("the command failed")
    assert not os.path.exists(path)


def test_names_cannot_carry_code():
    for name in ("V", "a'b", "a;b", "", "x" * 41):
        with pytest.raises(ValueError), powershell_values("1", {name: "x"}):
            pass


def test_osascript_values_are_arguments_and_a_dash_is_never_an_option():
    command = osascript_command("return item 1 of heliox_args", [HOSTILE, "-e", ""])
    assert command[:2] == ["osascript", "-e"] and HOSTILE not in command[2]
    assert command[3:] == ["x" + HOSTILE, "x-e", "x"]


@pytest.mark.skipif(sys.platform != "win32", reason="runs PowerShell")
@pytest.mark.parametrize("value", [HOSTILE, "", HOSTILE * 400], ids=["hostile", "empty", "long"])
def test_powershell_receives_the_value_whole(value):
    from pilot.system.platform_detect import run_powershell

    code, out, err = asyncio.run(run_powershell(ECHO, values={"v": value}, timeout=60))
    assert code == 0, err
    assert base64.b64decode(out.strip()).decode("utf-8") == value


@pytest.mark.skipif(sys.platform != "darwin", reason="runs osascript")
def test_osascript_receives_the_values_whole():
    done = subprocess.run(
        osascript_command('return (item 1 of heliox_args) & "|" & (item 2 of heliox_args)', [HOSTILE, "-e"]),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.rstrip("\n") == HOSTILE + "|-e"


def test_the_notification_text_is_data_on_every_system(monkeypatch):
    from pilot.system import notify, platform_detect

    seen: dict = {}

    async def fake_powershell(script, **kwargs):
        seen["windows"] = (script, kwargs.get("values"))
        return 0, "", ""

    async def fake_command(args, **_kwargs):
        seen.setdefault("commands", []).append(args)
        return 0, "", ""

    monkeypatch.setattr(platform_detect, "run_powershell", fake_powershell)
    monkeypatch.setattr(platform_detect, "run_command", fake_command)
    for platform in ("windows", "macos", "linux"):
        assert asyncio.run(notify.show(platform, HOSTILE, HOSTILE)) == (0, "")
    script, values = seen["windows"]
    assert values == {"summary": HOSTILE, "body": HOSTILE} and HOSTILE not in script
    mac, linux = seen["commands"]
    assert HOSTILE not in mac[2] and mac[3:] == ["x" + HOSTILE, "x" + HOSTILE]
    assert linux[linux.index("--") + 1 :] == [HOSTILE, HOSTILE]


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
def test_the_calendar_notification_carries_an_invitations_title_as_data(monkeypatch, system):
    """The calendar check's notification holds event titles, which whoever sends the invitation chooses."""
    from pilot.agents import email_agent

    calls: list[tuple[list[str], dict | None]] = []
    monkeypatch.setattr(email_agent.platform, "system", lambda: system)
    monkeypatch.setattr(
        email_agent.subprocess, "run", lambda args, **kwargs: calls.append((list(args), kwargs.get("env")))
    )
    title = f'"{HOSTILE}" has no video-call link.'
    email_agent._send_os_notification("Missing Meeting Link", title)
    ((args, env),) = calls
    assert all(HOSTILE not in arg for arg in args[:3])
    if system == "Windows":
        assert env["HELIOX_ARG_BODY"] == title
    elif system == "Darwin":
        assert args[-1] == "x" + title
    else:
        assert args[args.index("--") + 2] == title


def test_the_helper_module_keeps_its_promises():
    assert script_values.POWERSHELL[0] == "powershell"
