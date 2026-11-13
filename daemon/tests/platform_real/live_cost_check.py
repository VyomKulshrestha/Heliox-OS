"""What Heliox Live costs to keep running, on a real daemon (CI runners: Linux, macOS, Windows).

Starts a real daemon (fake home, no model) with Live on, reading typed text, no screen reading (OCR) and no
listening; lets it settle; leaves it idle for five minutes; then feeds it typing for five minutes through the readers'
fixture (a sentence every few seconds, some with a wrong weekday, so Live has something to raise). Prints INFO lines:

- resident memory (MB) of the daemon process, and of the daemon with everything it started (the Windows reader's
  PowerShell helper among them), at the start, at the end of the idle phase and at the end of the typing phase;
- CPU time (s, user + system) the daemon and its tree used in each phase;
- how many Live cards the typing raised (on the connection), so a phase in which Live read nothing is visible.

One CHECK: ``live-cost/no-runaway-growth``. It fails only when resident memory is still climbing through the last
two minutes of the idle phase: more than max(10 MB, 2%) higher at the end than at three minutes, and higher at every
one of the last four samples than the one before. No absolute figure is a pass or a fail: none has been chosen.

A hosted CI runner is not a person's laptop (other load, other memory, a virtual display); the numbers say what this
daemon costs there, not on an 8 GB laptop.

    python tests/platform_real/live_cost_check.py [--python /usr/bin/python3] [--idle 300] [--typing 300]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from ci_only import require_runner

require_runner("starts a real daemon with Live on and types into a window for ten minutes")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

import psutil  # noqa: E402
import real_daemon  # noqa: E402

real_daemon.time_limit("live-cost", 1500)

LIVE = "\n[live]\nenabled = true\nread_focused_text = true\nocr_fallback = false\nlisten_to_speech = false\n"
SAMPLE_SECONDS = 15.0
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)


def tree(process: psutil.Process) -> list[psutil.Process]:
    try:
        return [process, *process.children(recursive=True)]
    except psutil.Error:
        return [process]


def rss_mb(processes: list[psutil.Process]) -> float:
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            pass
    return round(total / 2**20, 1)


def cpu_s(processes: list[psutil.Process]) -> float:
    total = 0.0
    for process in processes:
        try:
            times = process.cpu_times()
            total += times.user + times.system
        except psutil.Error:
            pass
    return total


def sentences():
    """A sentence at a time, every third one with a weekday that does not match its date."""
    n = 0
    while True:
        n += 1
        if n % 3 == 0:
            day = date.today() + timedelta(days=5 + n % 20)
            wrong = (day + timedelta(days=1)).strftime("%A")
            yield f"The review moved to {wrong} {day.day} {day.strftime('%B')}. "
        else:
            yield f"Notes for item {n}: check the figures and send them on. "


class Typist:
    """Types into the readers' fixture on this system; ``close`` ends it. ``how`` says which route."""

    def __init__(self, python: str, folder: Path) -> None:
        self.how = "none"
        self._close = lambda: None
        self.type = lambda text: None
        if sys.platform.startswith("linux"):
            from linux_atspi_check import Fixture, typer

            fixture = Fixture(python)
            if fixture.wait_for("ready"):
                self.type, self.how = typer(False, fixture)
            self._close = fixture.close
        elif sys.platform == "darwin":
            fixture = subprocess.Popen(
                [sys.executable, str(HERE / "live_fixture_macos.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True,
            )  # fmt: skip
            assert fixture.stdin is not None and fixture.stdout is not None
            if fixture.stdout.readline().startswith("ready"):
                import threading

                import mac_front

                threading.Thread(target=lambda: list(fixture.stdout), daemon=True).start()

                def send(command: str) -> None:
                    try:
                        fixture.stdin.write(command + "\n")
                        fixture.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass

                def type_text(text: str) -> None:
                    mac_front.bring_to_front(fixture.pid)
                    send("front")
                    for word in text.split(" "):
                        send(f"insert {word} ")
                        time.sleep(0.15)

                self.type, self.how = type_text, "inside the app (no input tool on the runner)"

                def close() -> None:
                    send("quit")
                    mac_front.end(fixture)

                self._close = close
            else:
                fixture.kill()
        else:
            import pyautogui
            import windows_fields_check as W

            program = next((p for p in W.BROWSERS["edge"] if Path(p).exists()), None)
            if program is not None:
                page = folder / "field_fixture.html"
                page.write_text((HERE / "field_fixture.html").read_text(encoding="utf-8"), encoding="utf-8")
                edge = subprocess.Popen(
                    [program, f"--user-data-dir={folder / 'profile'}", "--no-first-run", "--no-default-browser-check",
                     "--new-window", page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )  # fmt: skip
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and not W.windows_titled(W.TITLE):
                    time.sleep(0.5)
                shown = W.windows_titled(W.TITLE)
                if shown:
                    W.bring_to_front(shown[0][0])
                    time.sleep(2.0)

                    def type_text(text: str) -> None:
                        W.bring_to_front(shown[0][0])
                        pyautogui.write(text, interval=0.03)

                    self.type, self.how = type_text, "Edge, pyautogui"
                self._close = lambda: W.end_tree(edge.pid)

    def close(self) -> None:
        try:
            self._close()
        except Exception as exc:
            print(f"DIAG closing the fixture: {type(exc).__name__}: {exc}", flush=True)


async def run(args: argparse.Namespace) -> int:
    real_daemon.write_config(LIVE)
    daemon = real_daemon.start()
    session = None
    typist = None
    try:
        session = await real_daemon.connect(240)
        ready, _status = await session.wait_ready(240)
        if not ready:
            check("live-cost/daemon-ready", False, "the daemon did not get ready")
            return 1
        status = await session.call("live_status")
        print(f"INFO live-cost: Live on {status.get('enabled')}; reads focused text {status.get('reads_focused_text')}")
        served = real_daemon.serving_process(daemon)
        main = psutil.Process(served.pid) if served is not None else psutil.Process(daemon.pid)
        await session.collect(30.0)  # settle: start-up work done

        start_rss, start_tree = rss_mb([main]), rss_mb(tree(main))
        cpu0, cpu0_tree = cpu_s([main]), cpu_s(tree(main))
        samples: list[tuple[float, float]] = []
        began = time.monotonic()
        while time.monotonic() - began < args.idle:
            await session.collect(SAMPLE_SECONDS)
            samples.append((time.monotonic() - began, rss_mb([main])))
        idle_rss, idle_tree = rss_mb([main]), rss_mb(tree(main))
        cpu1, cpu1_tree = cpu_s([main]), cpu_s(tree(main))

        with tempfile.TemporaryDirectory(prefix="heliox-live-cost-") as folder:
            typist = Typist(args.python, Path(folder))
            print(f"INFO live-cost: typing with {typist.how}", flush=True)
            cards_before = sum(1 for m in session.notifications if m.get("method") == "live_interjection")
            source = sentences()
            began = time.monotonic()
            while time.monotonic() - began < args.typing:
                await asyncio.to_thread(typist.type, next(source))
                await session.collect(4.0)  # a pause after each sentence, as people make
            cards = sum(1 for m in session.notifications if m.get("method") == "live_interjection") - cards_before
            typist.close()
            typist = None
        end_rss, end_tree = rss_mb([main]), rss_mb(tree(main))
        cpu2, cpu2_tree = cpu_s([main]), cpu_s(tree(main))

        print(
            f"INFO live-cost: system {sys.platform}; total memory {round(psutil.virtual_memory().total / 2**30, 1)} GB"
        )
        print(f"INFO live-cost: resident MB, daemon: start {start_rss}, after idle {idle_rss}, after typing {end_rss}")
        print(f"INFO live-cost: resident MB, with its helpers: start {start_tree}, after idle {idle_tree}, "
              f"after typing {end_tree}")  # fmt: skip
        print(f"INFO live-cost: CPU s, daemon: idle {cpu1 - cpu0:.1f} in {args.idle:.0f} s, typing {cpu2 - cpu1:.1f} "
              f"in {args.typing:.0f} s")  # fmt: skip
        print(f"INFO live-cost: CPU s, with its helpers: idle {cpu1_tree - cpu0_tree:.1f}, typing "
              f"{cpu2_tree - cpu1_tree:.1f}")  # fmt: skip
        print(f"INFO live-cost: Live cards raised while typing: {cards}", flush=True)

        late = [rss for at, rss in samples if at >= args.idle - 120]
        at_three = next((rss for at, rss in samples if at >= args.idle - 120), start_rss)
        rising = len(late) >= 5 and all(b > a for a, b in zip(late[-5:], late[-4:], strict=False))
        grew = idle_rss - at_three
        runaway = rising and grew > max(10.0, 0.02 * at_three)
        check(
            "live-cost/no-runaway-growth",
            not runaway,
            f"idle samples (s, MB) {[(round(a), b) for a, b in samples]}; last two minutes grew {grew:.1f} MB",
        )
    finally:
        if typist is not None:
            typist.close()
        await real_daemon.stop(daemon, session)
    return 0 if results and all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--idle", type=float, default=300.0)
    parser.add_argument("--typing", type=float, default=300.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
