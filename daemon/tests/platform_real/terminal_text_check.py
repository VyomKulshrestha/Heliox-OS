"""A real run of Live's kitty and WezTerm reader (pilot.live.terminal_text), for the ``platform-readers`` job on Linux.

Under an X display, each terminal that is installed is started with a plain ``sh`` (prompt ``$ ``) in a temporary
folder; ``git push origin main --force`` is typed into it through the terminal's own send-text (no Enter), and the
reader must return exactly that line, the way Live would see it. For kitty the config Heliox reads is the one in
``KITTY_CONFIG_DIRECTORY`` (remote control on a Unix socket); without it the reader must ask nothing. Each check
prints one ``CHECK`` line (``SKIP`` when a terminal is not installed); the exit code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.live.terminal_text import KITTY_SETUP, TerminalText  # noqa: E402

RISKY = "git push origin main --force"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def skip(name: str, why: str) -> None:
    print(f"CHECK {name}: SKIP - {why}", flush=True)


async def read_until(reader: TerminalText, terminal: str, pid: int | None, seconds: float = 15.0):
    end = time.monotonic() + seconds
    seen = None
    while time.monotonic() < end:
        seen = await reader.refresh(terminal, pid)
        if seen is not None and seen.line == RISKY:
            return seen
        await asyncio.sleep(0.5)
    return seen


async def kitty_checks(root: Path, env: dict[str, str]) -> None:
    kitty = shutil.which("kitty")
    if not kitty:
        skip("kitty", "kitty is not installed")
        return
    plain = root / "kitty-plain"
    plain.mkdir()
    (plain / "kitty.conf").write_text("font_size 10\n", encoding="utf-8")
    refused = TerminalText(env={"KITTY_CONFIG_DIRECTORY": str(plain)})
    asked: list[list[str]] = []

    async def recording(argv, timeout):
        asked.append(argv)
        return 1, ""

    refused._run = recording
    await refused.refresh("kitty", 1)
    check("kitty: without remote control nothing is asked", asked == [] and refused.problem.get("kitty") == KITTY_SETUP)

    conf_dir = root / "kitty-conf"
    conf_dir.mkdir()
    listen = f"unix:{root}/hlx-kitty"
    (conf_dir / "kitty.conf").write_text(
        f"allow_remote_control socket-only\nlisten_on {listen}\nconfirm_os_window_close 0\n", encoding="utf-8"
    )
    run_env = dict(env, KITTY_CONFIG_DIRECTORY=str(conf_dir), PS1="$ ", ENV="", LIBGL_ALWAYS_SOFTWARE="1")
    proc = subprocess.Popen([kitty, "--config", str(conf_dir / "kitty.conf"), "sh"], env=run_env, cwd=root)
    try:
        address = f"{listen}-{proc.pid}"
        socket = Path(address.removeprefix("unix:"))
        for _ in range(60):
            if socket.exists():
                break
            time.sleep(0.5)
        if not check("kitty: started with remote control on", socket.exists(), address):
            return
        time.sleep(1.0)
        subprocess.run([kitty, "@", "--to", address, "send-text", RISKY], timeout=10, check=False)
        reader = TerminalText(env={"KITTY_CONFIG_DIRECTORY": str(conf_dir)})
        seen = await read_until(reader, "kitty", proc.pid)
        check(
            "kitty: the line being typed is read, after the prompt", seen is not None and seen.line == RISKY, repr(seen)
        )
        seen_any = await read_until(TerminalText(env={"KITTY_CONFIG_DIRECTORY": str(conf_dir)}), "kitty", None, 5.0)
        check(
            "kitty: found without being told its pid", seen_any is not None and seen_any.line == RISKY, repr(seen_any)
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)


async def wezterm_checks(root: Path, env: dict[str, str]) -> None:
    wezterm = shutil.which("wezterm")
    if not wezterm:
        skip("wezterm", "wezterm is not installed")
        return
    run_env = dict(env, PS1="$ ", ENV="", LIBGL_ALWAYS_SOFTWARE="1")
    proc = subprocess.Popen(
        [wezterm, "--config", "front_end='Software'", "start", "--always-new-process", "--cwd", str(root), "sh"],
        env=run_env,
    )
    try:
        time.sleep(4.0)
        sent = subprocess.run([wezterm, "cli", "send-text", "--no-paste", RISKY], timeout=10, check=False, env=run_env)
        check("wezterm: started and took the text", sent.returncode == 0)
        seen = await read_until(TerminalText(env=run_env), "wezterm", None)
        check(
            "wezterm: the line being typed is read, after the prompt",
            seen is not None and seen.line == RISKY,
            repr(seen),
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)


async def main() -> int:
    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        skip("terminal text", "needs Linux with an X display")
        return 0
    with tempfile.TemporaryDirectory(prefix="hlx-term-") as folder:
        root = Path(folder)
        env = dict(os.environ)
        await kitty_checks(root, env)
        await wezterm_checks(root, env)
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
