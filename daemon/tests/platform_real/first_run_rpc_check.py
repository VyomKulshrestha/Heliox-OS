"""After a first-run setup or an update: does the daemon that was just installed answer its RPC and finish starting?

Run with the new environment's Python while that daemon starts. Waits for the auth token the daemon writes,
connects to ws://127.0.0.1:8785 and authenticates, then sends requests and waits for the reply with the matching
id, skipping notifications (since start-up became observable, the first frame is a ``startup_status`` notice, not
an answer). Checks that ``ping`` answers, then polls ``startup_status`` until its state is "ready" (start-up finished). Prints CHECK
lines; exit code 0 when both passed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


async def request(ws, method: str, request_id: int, timeout: float = 15.0) -> dict:
    """Send one request and return the reply carrying its id; notifications in between are skipped."""
    await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": {}, "id": request_id}))
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no reply to {method}")
        message = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        if message.get("id") == request_id:
            return message


HEAVY = ("scipy", "sklearn", "pandas", "river", "torch")


def timing_checks(label: str, timings: dict, late_parts: dict) -> tuple[list[str], bool]:
    """CHECK lines from the daemon's own start-up timings (pilot.system.startup_timing), and whether all passed:
    the times are the daemon's, measured from its process start, not this checker's polling."""
    lines: list[str] = []
    ok = True
    for point, words in (("listening", "listening"), ("ready", "ready"), ("all_started", "every late part started")):
        found = timings.get(point)
        if found is None:
            lines.append(f"CHECK start-up/{label}/{point}: FAIL - never reached {words}")
            ok = False
            continue
        lines.append(
            f"CHECK start-up/{label}/{point}: PASS - {words} {found['seconds']:.1f} s after {found['since']}, "
            f"{found['modules']} modules loaded"
        )
    if "learner_loaded" in timings:
        lines.append(f"CHECK start-up/{label}/learner: PASS - loaded {timings['learner_loaded']['seconds']:.1f} s")
    ready = timings.get("ready") or {}
    heavy = [name for name in ready.get("heavy", []) if name in HEAVY]
    if heavy:
        lines.append(f"CHECK start-up/{label}/nothing-heavy-before-ready: FAIL - loaded at ready: {', '.join(heavy)}")
        ok = False
    elif ready:
        lines.append(f"CHECK start-up/{label}/nothing-heavy-before-ready: PASS - none of {', '.join(HEAVY)} at ready")
    failed = sorted(name for name, state in late_parts.items() if state != "ready")
    if failed:
        lines.append(f"CHECK start-up/{label}/late-parts: FAIL - not started: {', '.join(failed)}")
        ok = False
    return lines, ok


# The parts every planned request reads, and a request a window or a person would send for each (status RPCs that
# need the part). Sent about a second after ready, each on its own connection, to show how long it waits.
ON_DEMAND = (
    ("screen_vision", "screen_vision_stats"),
    ("subconscious", "subconscious_stats"),
    ("strategy_evolution", "strategy_evolution_status"),
)
ON_DEMAND_BOUND = 5.0  # pilot.system.late_start.LATE_PART_WAIT_SECONDS
ON_DEMAND_MARGIN = 2.0


async def ask_on_demand(token: str, label: str, ready_seen_at: float) -> list[str]:
    """Each of the three most-used parts asked for at once, about 1 s after ready: how long each request waited and
    what it was answered (``ok``: started out of turn and ready; ``starting``: still starting after the bound)."""
    import websockets

    await asyncio.sleep(max(0.0, ready_seen_at + 1.0 - time.monotonic()))
    sent_after = time.monotonic() - ready_seen_at

    async def one(part: str, method: str) -> str:
        async with websockets.connect("ws://127.0.0.1:8785", open_timeout=5) as ws:
            await request_auth(ws, token)
            began = time.monotonic()
            reply = await request(ws, method, 2, timeout=60)
            waited = time.monotonic() - began
        result = reply.get("result") if isinstance(reply.get("result"), dict) else {}
        answer = result.get("status") or ("error" if "error" in reply else "answered")
        ok = waited <= ON_DEMAND_BOUND + ON_DEMAND_MARGIN and "error" not in reply
        return (
            f"CHECK start-up/{label}/on-demand/{part}: {'PASS' if ok else 'FAIL'} - {method} sent {sent_after:.1f} s "
            f"after ready waited {waited:.2f} s (bound {ON_DEMAND_BOUND:g} s) and was answered {answer!r}"
        )

    return list(await asyncio.gather(*(one(part, method) for part, method in ON_DEMAND)))


def part_lines(label: str, times: dict, ready_at: float | None) -> list[str]:
    """When each part that starts after ready began and settled, in seconds after process start."""
    lines = []
    for name, points in sorted(times.items(), key=lambda item: item[1].get("started", 1e9)):
        started, settled = points.get("started"), points.get("settled")
        after = f", {settled - ready_at:.1f} s after ready" if settled is not None and ready_at is not None else ""
        lines.append(
            f"INFO start-up/{label}/part {name}: started {started if started is not None else '-'} s, "
            f"settled {settled if settled is not None else '-'} s after process start{after}"
        )
    return lines


def package_lines(label: str, packages: dict) -> list[str]:
    """What was loaded at ready, by package, kept for the next start's comparison; after a later start, the packages
    one start had at ready and the other did not (the first run loads about 200 more modules than a saved state)."""
    import os
    from pathlib import Path

    folder = Path(os.environ.get("RUNNER_TEMP") or ".")
    (folder / f"startup-packages-{label}.json").write_text(json.dumps(packages), encoding="utf-8")
    if label == "first-run":
        return []
    try:
        first = json.loads((folder / "startup-packages-first-run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    diff = {name: first.get(name, 0) - packages.get(name, 0) for name in set(first) | set(packages)}
    shown = sorted((item for item in diff.items() if item[1]), key=lambda item: -abs(item[1]))[:25]
    return [
        f"INFO start-up/{label}/modules at ready, first run minus this start: {count:+d} {name}"
        for name, count in shown
    ]


def report_daemon(lines: int = 80) -> None:
    """When the daemon does not answer: its own log (the file, flushed per line, unlike its redirected stdout), the
    last start-up stage it reached, so a CI failure says where it stopped."""
    try:
        from pilot.config import LOG_FILE

        text = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as error:
        print(f"DAEMON-LOG unavailable: {type(error).__name__}: {error}")
        return
    marks = ("Start-up:", "Start-up timing", "Pilot daemon ready", "Learner loaded", "credential store")
    stages = [line for line in text if any(mark in line for mark in marks)]
    print(f"DAEMON-STAGE {stages[-1].strip() if stages else 'no start-up stage logged'}")
    print(f"DAEMON-LOG last {min(lines, len(text))} lines of {LOG_FILE}:")
    for line in text[-lines:]:
        print(f"  {line}")


def keyring_backend() -> str:
    """The credential store this environment would use (PYTHON_KEYRING_BACKEND decides when set)."""
    import os

    chosen = os.environ.get("PYTHON_KEYRING_BACKEND", "")
    try:
        import keyring

        return f"{type(keyring.get_keyring()).__name__} (PYTHON_KEYRING_BACKEND={chosen!r})"
    except Exception as error:
        return f"unavailable ({type(error).__name__}; PYTHON_KEYRING_BACKEND={chosen!r})"


async def main(timeout: float, timings_label: str = "") -> int:
    import websockets

    from pilot.config import RUNTIME_DIR

    print(f"INFO credential store for this start: {keyring_backend()}")

    token_file = RUNTIME_DIR / "auth_token"
    started = time.monotonic()
    deadline = started + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if token_file.exists():
            try:
                async with websockets.connect("ws://127.0.0.1:8785", open_timeout=5) as ws:
                    token = token_file.read_text(encoding="utf-8").strip()
                    auth = await request_auth(ws, token)
                    pong = await request(ws, "ping", 2)
                    ok = "error" not in auth and "result" in pong
                    print(
                        f"CHECK first-run/daemon-answers: {'PASS' if ok else 'FAIL'} - {pong} after {time.monotonic() - started:.0f} s"
                    )
                    if not ok:
                        report_daemon()
                        return 1
                    ready = {}
                    request_id = 3
                    while time.monotonic() < deadline:
                        ready = await request(ws, "startup_status", request_id)
                        request_id += 1
                        if (ready.get("result") or {}).get("state") == "ready":
                            ready_seen_at = time.monotonic()
                            print(f"CHECK first-run/daemon-ready: PASS - after {time.monotonic() - started:.0f} s")
                            if not timings_label:
                                return 0
                            for line in await ask_on_demand(token, timings_label, ready_seen_at):
                                print(line)
                            # Then wait for every late part, and report the daemon's own times.
                            status = ready.get("result") or {}
                            while time.monotonic() < deadline and "all_started" not in status.get("timings", {}):
                                await asyncio.sleep(2)
                                status = (await request(ws, "startup_status", request_id)).get("result") or {}
                                request_id += 1
                            lines, ok = timing_checks(
                                timings_label, status.get("timings") or {}, status.get("late_parts") or {}
                            )
                            timings = status.get("timings") or {}
                            lines += part_lines(
                                timings_label,
                                status.get("late_part_times") or {},
                                (timings.get("ready") or {}).get("seconds"),
                            )
                            lines += package_lines(timings_label, (timings.get("ready") or {}).get("packages") or {})
                            for line in lines:
                                print(line)
                            if not ok:
                                report_daemon()
                            return 0 if ok else 1
                        await asyncio.sleep(0.2 if timings_label else 2)  # ready seen within 0.2 s when timed
                    print(f"CHECK first-run/daemon-ready: FAIL - still {ready} after {timeout:.0f} s")
                    report_daemon()
                    return 1
            except Exception as exc:  # not listening yet
                last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(2)
    print(
        f"CHECK first-run/daemon-answers: FAIL - no answer in {timeout:.0f} s ({last_error or 'no auth token written'})"
    )
    report_daemon()
    return 1


async def request_auth(ws, token: str) -> dict:
    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": token}, "id": 1}))
    deadline = time.monotonic() + 10
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), max(0.1, deadline - time.monotonic())))
        if message.get("id") == 1:
            return message


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--timings=")]
    label = next((argument.split("=", 1)[1] for argument in sys.argv[1:] if argument.startswith("--timings=")), "")
    sys.exit(asyncio.run(main(float(arguments[0]) if arguments else 180.0, label)))
