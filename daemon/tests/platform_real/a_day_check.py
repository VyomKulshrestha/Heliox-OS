"""A day in the life on a real Linux desktop (CI: the a-day workflow, ubuntu-24.04, nightly and by hand).

The pytest day (tests/test_a_day_in_the_life.py) runs the whole day on fake clocks. This runs the parts only a real
desktop shows, from the same day file (tests/data/a_day_in_the_life.json), against a real daemon (fake home, no
model) with Live on (reading typed text; no OCR, no listening):

1. each slip the day types goes, through xdotool, into the readers' GTK field under Xvfb, read by the real AT-SPI
   reader in the daemon: the slips marked welcome raise a Live card of that kind on the window's connection, the
   decoy raises none (``a-day/typed-<n>``);
2. a reminder set for one minute later, the window closed and the daemon frozen past it (the lid closed), then woken
   and a window opened: the reminder reaches that window exactly once, saying when it was due or that it was said
   (``a-day/lid``);
What left the machine is the pytest day's (a stub cloud model's request bodies); here the daemon's model address
refuses at once, so nothing can leave.

What it cannot show: the time of day (a real daemon runs on the real clock, so "after six" and the offers' hours are
the pytest day's), a real voice, or a person's judgement. Prints CHECK lines; exit code 1 if one fails.

    python tests/platform_real/a_day_check.py --python /usr/bin/python3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from ci_only import require_runner

require_runner("starts a real daemon, types into a window and freezes the daemon")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

import real_daemon  # noqa: E402

real_daemon.time_limit("a-day", 1200)

DAY = json.loads((HERE.parents[0] / "data" / "a_day_in_the_life.json").read_text(encoding="utf-8"))
LIVE = "\n[live]\nenabled = true\nread_focused_text = true\nocr_fallback = false\nlisten_to_speech = false\n"
FROZEN = 100.0  # past the reminder's minute, as in the frozen-daemon check
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)


def cards(session) -> list[dict]:
    return [m.get("params") or {} for m in session.notifications if m.get("method") == "live_interjection"]


async def typed(session, python: str) -> None:
    from linux_atspi_check import Fixture, typer

    fixture = Fixture(python)
    try:
        if not fixture.wait_for("ready"):
            check("a-day/fixture", False, "the GTK fixture did not start")
            return
        type_keys, how = typer(False, fixture)
        print(f"INFO a-day: typing with {how}", flush=True)
        type_keys("Notes: ")
        await session.collect(3.0)
        for number, entry in enumerate(e for e in DAY["typed"] if not e.get("after_hours")):
            before = len(cards(session))
            text = " ".join(step["type"] for step in entry["steps"] if "type" in step)
            type_keys(text + " ")
            await session.collect(15.0)  # the pause after it: Live speaks up at a pause
            got = [c.get("kind") for c in cards(session)[before:]]
            welcome = entry["welcome"]
            ok = all(kind in got for kind in welcome) and all(kind in welcome for kind in got)
            check(f"a-day/typed-{number}", ok, f"{entry['name']}: raised {got}, welcome {welcome}")
    finally:
        fixture.close()


async def lid(daemon) -> None:
    session = await real_daemon.connect(60)
    added = await session.call("reminder_add", {"text": DAY["lid"]["reminder"], "in_minutes": 1})
    await session.close()  # no window while the lid is closed
    if added.get("status") != "ok":
        check("a-day/lid", False, f"the reminder was not set: {added}")
        return
    await real_daemon.freeze(daemon, FROZEN)  # the lid closed: the daemon and all it started stop running
    await asyncio.sleep(5.0)  # the wake is noticed and the overdue tick runs
    window = await real_daemon.connect(60)
    await window.collect(10.0)
    mine = [
        m.get("params") or {}
        for m in window.notifications
        if m.get("method") in ("reminder", "reminders_missed") and DAY["lid"]["reminder"] in json.dumps(m)
    ]
    message = str(mine[0].get("message", "")) if mine else ""
    check(
        "a-day/lid",
        len(mine) == 1 and message.startswith(("This was due", "I said this")),
        f"{len(mine)} notification(s) on the first window after the wake: {message!r}",
    )
    await window.close()


async def run(python: str) -> int:
    real_daemon.write_config(LIVE)
    daemon = real_daemon.start()
    session = None
    try:
        session = await real_daemon.connect(240)
        ready, _status = await session.wait_ready(240)
        if not ready:
            check("a-day/ready", False, "the daemon did not get ready")
            return 1
        await typed(session, python)
        await session.close()
        session = None
        await lid(daemon)
    finally:
        await real_daemon.stop(daemon, session)
    return 0 if results and all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/usr/bin/python3")
    return asyncio.run(run(parser.parse_args().python))


if __name__ == "__main__":
    sys.exit(main())
