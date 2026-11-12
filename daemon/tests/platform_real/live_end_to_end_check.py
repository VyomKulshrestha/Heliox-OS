"""Heliox Live end to end on real macOS and Linux, for the ``platform-readers`` CI job.

The reader checks prove the field is read; this proves that a slip typed into a real window becomes a Live card.
Everything is the daemon's own code: ``PilotServer._live_observe`` (the foreground window, ``never_read`` for
excluded apps, the platform's focused-text reader, the password refusal) feeds a real ``LiveCompanion`` with the
default settings and Live turned on, ticked at its own pace. Only its speech is a silent stub.

Against the GTK fixture (X11 under Xvfb, or Wayland under sway with ``--wayland``) or the AppKit fixture on macOS:

1. the fixture's app is excluded: a sentence with a wrong weekday typed into its field gives no card;
2. the exclusion is lifted and a second such sentence is typed: a ``weekday_date`` card with that sentence's
   weekday and date as evidence;
3. a third one typed into the password field: no card;
4. the window is closed: the card is withdrawn with reason ``closed``.

Each step prints one ``CHECK`` line (``SKIP`` where the runner cannot read fields); the exit code is the number of
failed checks.

    python tests/platform_real/live_end_to_end_check.py [--wayland] [--python /usr/bin/python3]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from pilot.live import focused_text  # noqa: E402
from pilot.live.engine import TICK_SECONDS, LiveCompanion  # noqa: E402

failures = 0
CARD_WAIT = 20.0  # seconds Live may take to raise a card after the sentence is typed
QUIET_WAIT = 6.0  # seconds a "nothing happens" step is watched for


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def skip(name: str, why: str) -> None:
    print(f"CHECK {name}: SKIP - {why}", flush=True)


def slip(days_ahead: int) -> tuple[str, str]:
    """A sentence whose weekday is wrong for its date, and the evidence Live should quote."""
    day = date.today() + timedelta(days=days_ahead)
    wrong = (day + timedelta(days=1)).strftime("%A")
    evidence = f"{wrong} {day.day} {day.strftime('%B')}"
    return f"Lunch with Sam is on {evidence}, see you there. ", evidence


class Live:
    """A real LiveCompanion fed by the daemon's own observe method, with the settings the daemon starts with."""

    def __init__(self, folder: Path) -> None:
        from pilot.config import PilotConfig
        from pilot.server import PilotServer

        self.config = PilotConfig()
        self.config.live.enabled = True
        self.config.live.ocr_fallback = False
        self.events: list[tuple[str, dict[str, Any]]] = []
        reader = focused_text.create_reader()
        self.reader = reader
        server = SimpleNamespace(config=self.config, _live_reader=reader, _live_last_ocr=0.0)
        self.server = server

        async def observe():
            return await PilotServer._live_observe(server)  # type: ignore[arg-type]

        async def emit(event: str, payload: dict[str, Any]) -> None:
            self.events.append((event, payload))

        async def speak(_text: str, _urgent: bool) -> str:
            return "suppressed"  # never a sound on the runner

        self.engine = LiveCompanion(
            settings=lambda: self.config.live,
            observe=observe,
            emit=emit,
            speak=speak,
            relevance_path=folder / "relevance.json",
            timing_path=folder / "timing.json",
            on_disable=reader.close,
        )

    async def diagnose(self, prefix: str) -> None:
        """Why no card: what the daemon's observe gives, what the reader gives (and, on macOS, the same read on the
        main thread), the reader's status, and what Live is holding."""
        from pilot.agents.screen_vision import _get_active_window

        def say(label: str, value: Any) -> None:
            print(f"DIAG {prefix} {label}: {str(value)[:500]}", flush=True)

        try:
            say("observe() now", await self.engine._observe())
        except Exception as exc:
            say("observe() raised", f"{type(exc).__name__}: {exc}")
        say("Live's last look", getattr(self.engine, "_last_obs", None))
        try:
            say("reader.read() (worker thread)", await self.reader.read())
        except Exception as exc:
            say("reader.read() raised", f"{type(exc).__name__}: {exc}")
        if sys.platform == "darwin":
            from pilot.live import focused_text_macos as M

            shared = M.shared()
            say("the same read on the main thread", shared.read() if shared is not None else "(no reader)")
            # What is in front, from the shared helper (frontmost pid, its focused window, the windows front to
            # back), against the fixture's pid.
            say("the fixture's pid", getattr(self, "fixture_pid", None))
            import mac_front

            mac_front.diag(f"{prefix}: when Live read nothing")
        try:
            say("reader.status()", self.reader.status().as_dict())
        except Exception as exc:
            say("reader.status() raised", f"{type(exc).__name__}: {exc}")
        say("foreground window", await asyncio.to_thread(_get_active_window))
        say("pending", {k: (v.correction.kind, v.correction.evidence) for k, v in self.engine._pending.items()})
        say("events", [(e, p.get("kind"), p.get("evidence")) for e, p in self.events][-8:])
        last = getattr(self.engine, "_last_obs", None)
        if last is not None and last.text:
            from pilot.live.detectors import detect

            ctx = self.engine._context(last, last.source)
            say("detectors on that text", [(c.kind, c.evidence) for c in detect(last.text, ctx)])

    def cards(self) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == "live_interjection"]

    def withdrawn(self) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == "live_interjection_withdrawn"]

    async def run(self, seconds: float, until=lambda: False) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            await self.engine.tick()
            if until():
                return True
            await asyncio.sleep(TICK_SECONDS)
        return bool(until())

    async def while_typing(self, type_text, text: str, seconds: float, until=lambda: False) -> bool:
        """Type in a thread while Live keeps ticking, as it does while a person types."""
        typing = asyncio.ensure_future(asyncio.to_thread(type_text, text))
        hit = await self.run(seconds, until)
        await typing
        return hit or bool(until())


STEPS = (
    "live-excluded-app-gives-no-card",
    "live-typed-slip-becomes-a-card",
    "live-password-field-gives-no-card",
    "live-card-withdrawn-when-window-closes",
)


async def first_read(prefix: str, live: Live, front, type_text, seconds: float = 15.0) -> bool:
    """Live must read the fixture's field through the daemon's own observe before any check here may pass: a "no
    card" proves nothing when Live read nothing at all. The window is asked forward (an earlier step can leave
    another app in front) and a neutral word is put in the field (an empty field gives nothing to read)."""
    await asyncio.to_thread(type_text, "Notes:")
    deadline = time.monotonic() + seconds
    obs = None
    while time.monotonic() < deadline:
        await asyncio.to_thread(front)
        await asyncio.sleep(0.5)
        try:
            obs = await live.engine._observe()
        except Exception as exc:
            obs = f"{type(exc).__name__}: {exc}"
        if getattr(obs, "source", "") == "focused_field" and getattr(obs, "editable", False):
            break
    ok = check(f"{prefix}/live-reads-the-fixture-field", getattr(obs, "source", "") == "focused_field", repr(obs))
    if not ok:
        await live.diagnose(prefix)
    return ok


async def steps(prefix: str, live: Live, type_text, focus, close, front=lambda: None) -> None:
    from pilot.agents.screen_vision import _get_active_window

    if not await first_read(prefix, live, front, type_text):
        for name in STEPS:
            skip(f"{prefix}/{name}", "Live read nothing from the fixture here, so these would prove nothing")
        return

    app, title = await asyncio.to_thread(_get_active_window)
    print(f"{prefix}: foreground app {app!r}, title {title!r}")

    # 1. the app excluded (before Live has seen it, as when the person set it up earlier): nothing typed there
    # reaches Live
    live.config.timeline.excluded_apps = [*live.config.timeline.excluded_apps, app]
    await live.run(2.0)
    first, first_evidence = slip(9)
    await live.while_typing(type_text, first, QUIET_WAIT, until=lambda: bool(live.cards()))
    check(f"{prefix}/live-excluded-app-gives-no-card", not live.cards(), repr(live.cards()[:1]))
    live.config.timeline.excluded_apps = [name for name in live.config.timeline.excluded_apps if name != app]

    # 2. the exclusion lifted: the next slip typed there is a card, quoting it
    await live.run(2.0)  # what is already there is the person's starting text, not a new slip
    second, evidence = slip(12)

    def card_for(text: str):
        return next((c for c in live.cards() if c.get("evidence") == text), None)

    await live.while_typing(type_text, second, CARD_WAIT, until=lambda: card_for(evidence) is not None)
    card = card_for(evidence)
    if not check(
        f"{prefix}/live-typed-slip-becomes-a-card",
        card is not None and card.get("kind") == "weekday_date" and card.get("source") == "focused_field",
        repr(card or live.cards()[-1:]),
    ):
        await live.diagnose(prefix)
    # The first sentence may now get its card too: once the app is no longer excluded, what its field holds may be
    # read, and the first field Live sees is checked whole (someone who turns Live on mid-draft wants it checked).
    print(
        f"{prefix}: cards so far {[c.get('evidence') for c in live.cards()]} (typed while excluded: {first_evidence})"
    )

    # 3. the same kind of slip in the password field: nothing
    await asyncio.to_thread(focus, "password")
    third, secret_evidence = slip(16)
    await live.while_typing(type_text, third, QUIET_WAIT, until=lambda: card_for(secret_evidence) is not None)
    check(
        f"{prefix}/live-password-field-gives-no-card",
        card_for(secret_evidence) is None,
        repr([c.get("evidence") for c in live.cards()]),
    )

    # 4. the window closed: the card goes with it
    if card is None:
        skip(f"{prefix}/live-card-withdrawn-when-window-closes", "no card to withdraw")
        return
    await asyncio.to_thread(close)

    def gone():
        return any(w.get("id") == card["id"] for w in live.withdrawn())

    await live.run(10.0, until=gone)
    reason = next((w.get("reason") for w in live.withdrawn() if w.get("id") == card["id"]), None)
    check(f"{prefix}/live-card-withdrawn-when-window-closes", reason == "closed", f"reason {reason!r}")


async def linux(wayland: bool, python: str, folder: Path) -> None:
    import atexit

    from linux_atspi_check import Fixture, typer

    prefix = "wayland" if wayland else "x11"
    fixture = Fixture(python)
    try:
        if not check(f"{prefix}/live-fixture-started", fixture.wait_for("ready")):
            return
        type_keys, how = typer(wayland, fixture)
        print(f"typing with: {how}")
        if wayland and shutil.which("wtype"):
            keeper = subprocess.Popen(["wtype", "-s", "600000"])  # gives the headless seat a keyboard
            atexit.register(keeper.kill)
            time.sleep(1.0)
        live = Live(folder)

        def focus(which: str) -> None:
            fixture.send(f"focus {which}")

        await steps(prefix, live, type_keys, focus, fixture.close, front=lambda: focus("entry"))
    finally:
        fixture.close()


def mac_front_for(pid: int, send) -> Callable[[], None]:
    """Bring a macOS fixture forward the way that works on the runner: activation asked by this (the check's) process
    through ``mac_front.bring_to_front``, which the fixture asking for itself is not (macOS 14's cooperative
    activation ignores it after the job's first check). The fixture's own "front" then makes its window key and its
    text field the first responder, which the helper does not do from outside. Prints the route whenever it
    changes."""
    import mac_front

    last = [None]

    def front() -> None:
        route = mac_front.bring_to_front(pid)
        if route != last[0]:
            print(f"macos: fixture pid {pid} brought forward by: {route or '(nothing worked)'}", flush=True)
            last[0] = route
        send("front")

    return front


async def macos(folder: Path) -> None:
    from pilot.live import focused_text_macos as M
    from pilot.system import macos_permissions as mp

    if mp.accessibility_state() != mp.PermissionState.GRANTED:
        skip("macos/live-end-to-end", "this runner does not grant Accessibility to the job, so no field can be read")
        return
    fixture = subprocess.Popen(
        [sys.executable, str(HERE / "live_fixture_macos.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert fixture.stdin is not None and fixture.stdout is not None
    said: list[str] = []

    def send(command: str) -> None:
        assert fixture.stdin is not None
        try:
            fixture.stdin.write(command + "\n")
            fixture.stdin.flush()
        except (BrokenPipeError, OSError):
            return
        time.sleep(0.6)

    def type_text(text: str) -> None:
        # No input tool may send keys to another app on the runner; insert from inside the app, a few words at a
        # time, so Live sees the sentence grow.
        for word in text.split(" "):
            send(f"insert {word} ")

    def close() -> None:
        send("quit")
        fixture.wait(timeout=10)

    import mac_front

    mac_front.diag("live end to end: start")
    try:
        if not check("macos/live-fixture-started", fixture.stdout.readline().startswith("ready")):
            return
        # The fixture answers each command; read its answers so its output never fills up.
        threading.Thread(target=lambda: said.extend(fixture.stdout), daemon=True).start()
        time.sleep(1.0)
        print(f"reader shared: {M.shared() is not None}")
        live = Live(folder)
        live.fixture_pid = fixture.pid
        front = mac_front_for(fixture.pid, send)
        await steps("macos", live, type_text, lambda which: send(f"focus {which}"), close, front=front)
    finally:
        mac_front.end(fixture)  # nothing this check started stays in front of the next one
        mac_front.diag("live end to end: end")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wayland", action="store_true")
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="heliox-live-e2e-") as folder:
        if sys.platform == "darwin":
            asyncio.run(macos(Path(folder)))
        elif sys.platform.startswith("linux"):
            asyncio.run(linux(args.wayland, args.python, Path(folder)))
        else:
            skip("live-end-to-end", "macOS and Linux only; Windows has its own reader checks")
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens a window on this desktop and types into it")
    sys.exit(main())
