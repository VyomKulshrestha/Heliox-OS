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
import threading
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


async def never_read_steps(prefix: str, live: E.Live, send, type_text, front=lambda: None) -> None:
    # Live must first read this fixture's field, or "never read" proves nothing (it read nothing at all).
    if not await E.first_read(prefix, live, front, type_text):
        for title in ("Bitwarden", "Heliox-OS"):
            skip(f"{prefix}/live-window-titled-{title}-is-never-read", "Live read nothing from the fixture here")
        return
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
        await never_read_steps(
            prefix, E.Live(folder / "never"), fixture.send, type_keys, front=lambda: fixture.send("focus entry")
        )
    finally:
        fixture.close()


# ── 1b. a GTK 4 window Live never reads ──

GTK4_TITLE = "Heliox Live GTK 4 window"


async def gtk4_never_read(python: str, folder: Path) -> None:
    """GTK 4 windows are recognised by their numeric role (the reader's 490947a), so a GTK 4 field has a window title:
    the rules by title (a password manager's name, Heliox's own window) now apply there. Shown for real here: Live
    reads the GTK 4 field and its title, then a GTK 4 window titled like a password manager or like Heliox is never
    read. Runs in the X11 step that has GTK 4 and a window manager (openbox)."""
    prefix = "x11-gtk4"
    proc = subprocess.Popen(
        [python, str(HERE / "live_gtk4_window.py"), GTK4_TITLE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "NO_AT_BRIDGE"},
    )
    assert proc.stdin is not None and proc.stdout is not None
    said: list[str] = []

    def send(command: str) -> None:
        try:
            proc.stdin.write(command + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return
        time.sleep(0.5)

    def type_text(text: str) -> None:
        for word in text.split(" "):
            send(f"insert {word} ")

    def front() -> None:
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "search", "--name", "Heliox", "windowactivate"], timeout=10, capture_output=True)
        send("focus")

    try:
        if not check(f"{prefix}/window-started", proc.stdout.readline().startswith("ready")):
            return
        threading.Thread(target=lambda: said.extend(proc.stdout), daemon=True).start()
        live = E.Live(folder / "gtk4")
        if not await E.first_read(prefix, live, front, type_text):
            for title in ("Bitwarden", "Heliox-OS"):
                skip(f"{prefix}/live-window-titled-{title}-is-never-read", "Live read nothing from the GTK 4 window")
            return
        seen = await live.engine._observe()
        check(f"{prefix}/the-window-title-is-read", getattr(seen, "title", "") == GTK4_TITLE, repr(seen))
        await never_read_steps(prefix, live, send, type_text, front)
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ── 1c. an app Live cannot see into: said once, quietly ──


async def blind_spot(python: str, folder: Path) -> None:
    """A GTK window started with NO_AT_BRIDGE=1 stays off the accessibility bus: the reader cannot see its field, and
    the window's app and title come only from the window manager (openbox in this step). Live must say so once, in its
    status and the pill's note, never as a card. The check shortens the 45 s Live waits to 15 s."""
    prefix = "x11-blind"
    env = dict(os.environ, NO_AT_BRIDGE="1")
    proc = subprocess.Popen(
        [python, str(HERE / "gtk_fixture.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        if not check(f"{prefix}/window-started", proc.stdout.readline().startswith("ready")):
            return
        threading.Thread(target=lambda: list(proc.stdout), daemon=True).start()
        if shutil.which("xdotool"):
            subprocess.run(
                ["xdotool", "search", "--sync", "--name", "Heliox AT-SPI fixture", "windowactivate"],
                timeout=15,
                capture_output=True,
            )
        live = E.Live(folder / "blind")
        live.server._live = live.engine  # the daemon's observe tells Live what it could not read
        live.engine.blind.after = 15.0
        seen = await live.engine._observe()
        if getattr(seen, "source", "") == "focused_field":
            skip(f"{prefix}/live-says-once-it-cannot-see-into-the-app", "the reader saw this window after all")
            return

        def notes():
            return [p.get("blind_note") for e, p in live.events if e == "live_state" and p.get("blind_note")]

        await live.run(30.0, until=lambda: bool(live.engine.blind.notices()))
        await live.run(10.0)  # and nothing more after it
        told = live.engine.blind.notices()
        check(
            f"{prefix}/live-says-once-it-cannot-see-into-the-app",
            len(told) == 1 and told[0]["sentence"].startswith("Live can't see what you type in"),
            repr(told),
        )
        check(
            f"{prefix}/said-quietly-never-as-a-card",
            len(set(notes())) == 1 and not live.cards(),
            f"notes {notes()!r}; cards {live.cards()!r}",
        )
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


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
            pass
        if self.proc.poll() is None:  # killed and waited for, so it stays in front of nothing
            self.proc.kill()
            self.proc.wait(timeout=5)


async def error_on_screen(prefix: str, python: str, folder: Path) -> None:
    window = TextWindow(python, "Heliox CI build log", ERROR_TEXT)
    try:
        if not check(f"{prefix}/error-window-shown", window.ready()):
            return
        if sys.platform == "darwin":
            import mac_front

            route = mac_front.bring_to_front(window.proc.pid)
            print(f"macos: error window pid {window.proc.pid} brought forward by: {route or '(nothing worked)'}")
        if shutil.which("xdotool") and os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            subprocess.run(["xdotool", "search", "--sync", "--name", "Heliox CI build log", "windowfocus"], timeout=15)
        live = E.Live(folder / "ocr")
        live.config.live.ocr_fallback = True  # screen reading on, as the person would turn it on
        # First read: the daemon's own observe must read the screen before a card can be judged (on Wayland through
        # the desktop's Screenshot portal). If it never does, say what the screen read gave and judge nothing.
        seen, deadline = None, time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                seen = await live.engine._observe()
            except Exception as exc:
                seen = f"{type(exc).__name__}: {exc}"
            if getattr(seen, "source", "") == "screen_ocr" and getattr(seen, "text", "").strip():
                break
            await asyncio.sleep(1.0)
            live.server._live_last_ocr = 0.0  # the daemon reads the screen at most every 5 s; ask again now
        if not check(f"{prefix}/live-reads-the-screen", getattr(seen, "source", "") == "screen_ocr", repr(seen)[:300]):
            await screen_diag()
            skip(f"{prefix}/live-error-on-screen-becomes-a-card", "the screen could not be read here")
            return
        live.server._live_last_ocr = 0.0

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
            await screen_diag()
    finally:
        window.close()


async def screen_diag() -> None:
    from pilot.system.vision import screen_ocr

    try:
        text = await screen_ocr()
    except Exception as exc:
        text = f"({type(exc).__name__}: {exc})"
    print(f"DIAG what the screen read gave: {text[:400]!r}", flush=True)


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
            [sys.executable, str(HERE / "live_fixture_macos.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert fixture.stdin is not None and fixture.stdout is not None
        said: list[str] = []

        def send(command: str) -> None:
            fixture.stdin.write(command + "\n")
            fixture.stdin.flush()
            time.sleep(0.6)

        def type_text(text: str) -> None:
            for word in text.split(" "):
                send(f"insert {word} ")

        import mac_front

        mac_front.diag("live more checks: start")
        try:
            if check("macos/never-read-fixture-started", fixture.stdout.readline().startswith("ready")):
                threading.Thread(target=lambda: said.extend(fixture.stdout), daemon=True).start()
                time.sleep(1.0)
                live = E.Live(folder / "never")
                live.fixture_pid = fixture.pid  # for the DIAG lines: whose window is in front
                await never_read_steps("macos", live, send, type_text, front=E.mac_front_for(fixture.pid, send))
        finally:
            mac_front.end(fixture)
        await error_on_screen("macos", sys.executable, folder)
        mac_front.diag("live more checks: end")
        skip("macos/live-back-and-forth", "nothing on the runner can give another process's window focus")
        return
    await linux_never_read(args.wayland, args.python, folder)
    if args.only == "blind-spot":
        await blind_spot(args.python, folder)
        return
    if args.only == "gtk4-never-read":
        await gtk4_never_read(args.python, folder)
        return
    if args.only == "error-on-screen":
        await error_on_screen("wayland" if args.wayland else "x11", args.python, folder)
        return
    if args.wayland:
        # Runs in the "Wayland portals" step instead (--only error-on-screen), where the Screenshot portal runs.
        skip("wayland/live-error-on-screen-becomes-a-card", "runs in the Wayland portals step, with the portal")
    else:
        await error_on_screen("x11", args.python, folder)
    await back_and_forth(args.wayland, args.python, folder)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wayland", action="store_true")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--only", choices=("all", "error-on-screen", "gtk4-never-read", "blind-spot"), default="all")
    args = parser.parse_args()
    if not (sys.platform == "darwin" or sys.platform.startswith("linux")):
        skip("live-more", "macOS and Linux only")
        return 0
    with tempfile.TemporaryDirectory(prefix="hlx-live-", dir="/tmp") as folder:
        root = Path(folder)
        for name in ("never", "ocr", "switch", "gtk4", "blind"):
            (root / name).mkdir()
        asyncio.run(run(args, root))
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens windows on this desktop and types into them")
    sys.exit(main())
