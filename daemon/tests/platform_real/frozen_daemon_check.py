"""A stand-in for sleep: the whole daemon frozen for minutes, then let go (CI runners: Linux, macOS, Windows).

A real sleep cannot be made on a runner. Freezing the daemon's process (SIGSTOP and SIGCONT; on Windows every thread
suspended, through psutil) gives the part of a sleep the daemon itself can see: its loops did not run for minutes, its
connections went quiet, and timers came due meanwhile. The check:

1. starts a real daemon (fake home), waits until it is ready and sets a reminder for one minute later;
2. freezes the daemon for FROZEN seconds (past the reminder and past the wake detector's threshold), then lets it go;
3. measures how long it takes to answer a ping again, on the old connection or a new one;
4. checks it noticed the gap as a wake ("Clock event: wake" in its log, pilot.system.wake), within a few seconds;
5. checks the reminder that came due while it was frozen went out once, to the first window, late and saying when it
   was due ("This was due at ...", or "I said this at ..." had the voice said it to a room with no window): not
   still waiting, not held for a window that never got it, not given twice. When this fails it prints the last 60
   lines of the daemon's log.

What it cannot show: the clocks of a real sleep (on Linux and macOS the monotonic clock stops during sleep; here
both run on), the wake phrase needed again, the network and audio devices going away and coming back, and the
window's own reconnection. Prints CHECK lines; exit code 1 if one fails. Off a runner it changes nothing (ci_only).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

from ci_only import require_runner

require_runner("starts, freezes and resumes a real daemon")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

import real_daemon  # noqa: E402

real_daemon.time_limit("frozen", 600)  # no check may hang: a CHECK line and exit 1 instead

FROZEN = 100.0  # seconds: past the reminder (60 s) and past WAKE_THRESHOLD_SECONDS (60 s) plus a check
ANSWER_BOUND = 10.0
WAKE_BOUND = 15.0  # the wake watch looks every 5 s
REMINDER_TEXT = "the frozen daemon check"
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"CHECK frozen/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    results.append(ok)
    return ok


def reminders_in(notifications: list[dict]) -> list[dict]:
    """The reminders among a connection's notifications: one "reminder", or each in a "reminders_missed" card."""
    found: list[dict] = []
    for message in notifications:
        params = message.get("params") or {}
        if message.get("method") == "reminder":
            found.append(params)
        elif message.get("method") == "reminders_missed":
            found.extend(item for item in params.get("reminders", []) if isinstance(item, dict))
    return found


async def held_reminders(limit: float) -> list[dict]:
    """Connect as a window does and collect the reminders that arrive within ``limit`` (a second window)."""
    session = await real_daemon.connect(60)
    try:
        return reminders_in(await session.collect(limit))
    finally:
        await session.close()


def held_now() -> list[dict]:
    """This check's reminder if the store still holds it for a window (``pending``; reminder_list does not list it)."""
    from pilot.config import DATA_DIR

    try:
        items = json.loads((DATA_DIR / "reminders.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [
        item for item in items if isinstance(item, dict) and item.get("pending") and REMINDER_TEXT in json.dumps(item)
    ]


async def run() -> int:
    real_daemon.write_config()
    daemon = real_daemon.start()
    session = None
    try:
        session = await real_daemon.connect(240)
        ready, _status = await session.wait_ready(240)
        added = await session.call("reminder_add", {"text": REMINDER_TEXT, "in_minutes": 1})
        if not check("set-up", ready and added.get("status") == "ok", f"ready: {ready}; reminder: {added}"):
            return 1
        await session.close()
        session = None

        await real_daemon.freeze(daemon, FROZEN)  # resumed whatever happens (real_daemon.freeze)
        resumed = time.monotonic()
        # No window for a moment, as after a real wake: the reminder that came due is held, not given to nobody.
        await asyncio.sleep(1.0)

        answered = None
        while time.monotonic() - resumed < 60 and answered is None:
            try:
                session = await real_daemon.connect(10)
                await session.call("ping", timeout=10)
                answered = time.monotonic() - resumed
            except Exception:
                await asyncio.sleep(0.5)
        check(
            "answers-again",
            answered is not None and answered <= ANSWER_BOUND,
            f"a ping answered {answered:.1f} s after the freeze ended" if answered is not None else "no answer in 60 s",
        )

        woke = None
        while time.monotonic() - resumed < 30 and woke is None:
            lines = real_daemon.log_lines(("Clock event: wake",), last=5)
            if lines:
                woke = time.monotonic() - resumed
            else:
                await asyncio.sleep(0.5)
        check(
            "noticed-the-wake",
            woke is not None and woke <= WAKE_BOUND,
            f"'Clock event: wake' logged {woke:.1f} s after the freeze ended" if woke is not None else "no wake logged",
        )

        # Nobody could see or hear the reminder when it came due (no window open, Heliox's voice not in use on a
        # runner), so it is held and the first window gets it, once, late, saying when it was due: "This was due at
        # 2:31 AM: ...". Had the voice said it to a room with no window, the window would get "I said this at ...".
        # A said-aloud reminder no longer counts as given (server._deliver_reminder, "unheard"), so neither does it
        # here: exactly one window, never zero.
        first = reminders_in(await session.collect(10)) if session is not None else []
        if session is not None:
            await session.close()
            session = None
        second = await held_reminders(5)  # it went out once: a later window does not get it again
        mine = [item for item in [*first, *second] if REMINDER_TEXT in json.dumps(item)]
        session = await real_daemon.connect(30)
        waiting = await session.call("reminder_list")
        still = [item for item in (waiting.get("reminders") or []) if REMINDER_TEXT in json.dumps(item)]
        held = held_now()  # reminder_list leaves out held ones: a reminder held but given to no window shows here
        late = [item for item in mine if item.get("late")]
        words = [str(item.get("message", "")) for item in mine]
        said_when = bool(words) and all(word.startswith(("This was due ", "I said this ")) for word in words)
        if not check(
            "reminder-went-out",
            len(mine) == 1 and len(late) == 1 and said_when and not still and not held,
            f"given {len(mine)} time(s) to a window: first window "
            f"{len([i for i in first if REMINDER_TEXT in json.dumps(i)])}, a later one "
            f"{len([i for i in second if REMINDER_TEXT in json.dumps(i)])}; late: {bool(late)} "
            f"({words[0][:80] if words else 'nothing shown'!r}); still waiting: {still}; held for a window: {held}",
        ):
            for line in real_daemon.log_tail(60):
                print(f"DAEMON-LOG {line}", flush=True)
    finally:
        await real_daemon.stop(daemon, session)
    return 0 if results and all(results) else 1


def main() -> int:
    """Every way out prints a CHECK line saying why: until 13 November 2026 check() returned nothing, so a passing
    set-up ended the run with exit code 1 and no other line (CI on 2f6f461, all three systems)."""
    try:
        code = asyncio.run(run())
    except BaseException as error:  # noqa: BLE001 - whatever ends the run is said, then it fails
        traceback.print_exc(file=sys.stdout)
        check("ran", False, f"{type(error).__name__}: {error}")
        return 1
    if code and all(results):
        check("ran", False, "the run ended with a failure that no line above names")
    return code


if __name__ == "__main__":
    sys.exit(main())
