"""More of Heliox Live on real macOS and Linux windows, for the ``platform-readers`` job (beside
``live_end_to_end_check.py``, whose real ``LiveCompanion`` fed by the daemon's own ``_live_observe`` this reuses).

1. **Windows Live never reads.** The fixture window retitled as a password manager ("Bitwarden") and as Heliox's
   own window ("Heliox-OS"): the daemon's observe returns nothing, and a wrong weekday typed there gives no card.
2. **An error on screen.** A window showing "fatal: not a git repository" in a large font, nothing to type in, with
   screen reading on: an ``error_on_screen`` card quoting it. (X11: tesseract and scrot; macOS: the Vision framework
   with Screen Recording; Wayland: skipped, the screenshot portal is not proven.)
3. **Back and forth.** Two fixture windows given focus in turn, ten times: the ``stuck`` offer to put them side by
   side. (X11: xdotool; sway: swaymsg; macOS: skipped, nothing here can give another process's window focus.)

Each step prints one ``CHECK`` line (``SKIP`` with the reason); the exit code is the number of failed checks.

    python tests/platform_real/live_more_checks.py [--wayland] [--python /usr/bin/python3]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import live_end_to_end_check as E  # noqa: E402

failures = 0
ERROR_TEXT = "fatal: not a git repository (or any of the parent directories): .git"


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def skip(name: str, why: str) -> None:
    print(f"CHECK {name}: SKIP - {why}", flush=True)


# ── 1. windows Live never reads ──


async def never_read_steps(prefix: str, live: E.Live, send, type_text) -> None:
    for days, title in ((20, "Bitwarden - My Vault"), (23, "Heliox-OS")):
        await asyncio.to_thread(send, f"title {title}")
        await live.run(1.5)
        seen = await live.engine._observe()
        text, evidence = E.slip(days)
        before = len(live.cards())
        await live.while_typing(type_text, text, E.QUIET_WAIT, until=lambda before=before: len(live.cards()) > before)
        cards = [c.get("evidence") for c in live.cards()[before:]]
        check(
            f"{prefix}/live-window-titled-{title.split()[0]}-is-never-read",
            seen is None and not cards,
            f"{seen!r} {cards}",
        )


async def linux_never_read(wayland: bool, python: str, folder: Path) -> None:
    import atexit

    from linux_atspi_check import Fixture, typer

    prefix = "wayland" if wayland else "x11"
    fixture = Fixture(python)
    try:
        if not check(f"{prefix}/never-read-fixture-started", fixture.wait_for("ready")):
            return
        type_keys, how = typer(wayland, fixture)
        print(f"typing with: {how}")
        if wayland and shutil.which("wtype"):
            keeper = subprocess.Popen(["wtype", "-s", "600000"])
            atexit.register(keeper.kill)
            time.sleep(1.0)
        await never_read_steps(prefix, E.Live(folder / "never"), fixture.send, type_keys)
    finally:
        fixture.close()


# ── 2. an error on screen ──


class TextWindow:
    def __init__(self, python: str, title: str, text: str) -> None:
        self.proc = subprocess.Popen(
            [python, str(HERE / "live_text_window.py"), title, text],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "NO_AT_BRIDGE"},
        )

    def ready(self, seconds: float = 30.0) -> bool:
        assert self.proc.stdout is not None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            line = self.proc.stdout.readline()
            if not line:
                return False
            if line.startswith("ready"):
                return True
        return False

    def close(self) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


async def error_on_screen(prefix: str, python: str, folder: Path) -> None:
    window = TextWindow(python, "Heliox CI build log", ERROR_TEXT)
    try:
        if not check(f"{prefix}/error-window-shown", window.ready()):
            return
        if shutil.which("xdotool") and os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            subprocess.run(["xdotool", "search", "--sync", "--name", "Heliox CI build log", "windowfocus"], timeout=15)
        live = E.Live(folder / "ocr")
        live.config.live.ocr_fallback = True  # screen reading on, as the person would turn it on

        def card():
            return next((c for c in live.cards() if c.get("kind") == "error_on_screen"), None)

        await live.run(25.0, until=lambda: card() is not None)
        found = card()
        check(
            f"{prefix}/live-error-on-screen-becomes-a-card",
            found is not None and "not a git repository" in str(found.get("evidence", "")),
            repr(found or [c.get("kind") for c in live.cards()]),
        )
        if found is None:
            from pilot.system.vision import screen_ocr

            try:
                text = await screen_ocr()
            except Exception as exc:
                text = f"({type(exc).__name__}: {exc})"
            print(f"DIAG what the screen read gave: {text[:400]!r}", flush=True)
    finally:
        window.close()


# ── 3. back and forth between two windows ──


def focus_by_title(title: str, wayland: bool) -> None:
    if wayland:
        socks = glob.glob(os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "sway-ipc.*.sock"))
        env = dict(os.environ, SWAYSOCK=socks[0]) if socks else None
        subprocess.run(["swaymsg", f'[title="{title}"] focus'], timeout=10, env=env, capture_output=True)
    else:
        subprocess.run(["xdotool", "search", "--sync", "--name", title, "windowfocus", "--sync"], timeout=10)


async def back_and_forth(wayland: bool, python: str, folder: Path) -> None:
    from linux_atspi_check import Fixture

    prefix = "wayland" if wayland else "x11"
    if (wayland and not shutil.which("swaymsg")) or (not wayland and not shutil.which("xdotool")):
        skip(f"{prefix}/live-back-and-forth", "no tool to give a window focus here")
        return
    left, right = Fixture(python), Fixture(python)
    try:
        if not check(f"{prefix}/two-fixtures-started", left.wait_for("ready") and right.wait_for("ready")):
            return
        names = ("Heliox CI left", "Heliox CI right")
        for fixture, name in zip((left, right), names, strict=True):
            fixture.send(f"title {name}")
            fixture.send(f"insert notes in {name.split()[-1]}")  # Live reads a field that has text in it
        live = E.Live(folder / "switch")
        seen: list[str] = []

        def offer():
            return next((c for c in live.cards() if c.get("kind") == "stuck"), None)

        for turn in range(12):
            await asyncio.to_thread(focus_by_title, names[turn % 2], wayland)
            await live.run(1.2, until=lambda: offer() is not None)
            obs = live.engine._last_obs
            seen.append(getattr(obs, "title", "") if obs is not None else "-")
            if offer() is not None:
                break
        # A card waits for the person to pause (moving focus counts as activity), so stop switching, as a person
        # would once they have what they came for, and give Live that pause.
        await live.run(8.0, until=lambda: offer() is not None)
        pending = [v.correction.kind for v in live.engine._pending.values()]
        check(
            f"{prefix}/live-back-and-forth-gives-the-side-by-side-offer",
            offer() is not None,
            f"titles seen {seen}; pending {pending}",
        )
    finally:
        left.close()
        right.close()


async def run(args: Any, folder: Path) -> None:
    if sys.platform == "darwin":
        from pilot.system import macos_permissions as mp

        if mp.accessibility_state() != mp.PermissionState.GRANTED:
            skip("macos/live-more", "this runner does not grant Accessibility to the job")
            return
        fixture = subprocess.Popen(
            [sys.executable, str(HERE / "macos_fixture.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        assert fixture.stdin is not None and fixture.stdout is not None

        def send(command: str) -> None:
            fixture.stdin.write(command + "\n")
            fixture.stdin.flush()
            time.sleep(0.6)

        def type_text(text: str) -> None:
            for word in text.split(" "):
                send(f"insert {word} ")

        try:
            if check("macos/never-read-fixture-started", fixture.stdout.readline().startswith("ready")):
                time.sleep(1.0)
                await never_read_steps("macos", E.Live(folder / "never"), send, type_text)
        finally:
            send("quit")
            fixture.kill()
        await error_on_screen("macos", sys.executable, folder)
        skip("macos/live-back-and-forth", "nothing on the runner can give another process's window focus")
        return
    await linux_never_read(args.wayland, args.python, folder)
    if args.wayland:
        skip("wayland/live-error-on-screen-becomes-a-card", "the Wayland screenshot portal is not proven yet")
    else:
        await error_on_screen("x11", args.python, folder)
    await back_and_forth(args.wayland, args.python, folder)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wayland", action="store_true")
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    if not (sys.platform == "darwin" or sys.platform.startswith("linux")):
        skip("live-more", "macOS and Linux only")
        return 0
    with tempfile.TemporaryDirectory(prefix="hlx-live-", dir="/tmp") as folder:
        root = Path(folder)
        for name in ("never", "ocr", "switch"):
            (root / name).mkdir()
        asyncio.run(run(args, root))
    return failures


if __name__ == "__main__":
    sys.exit(main())
