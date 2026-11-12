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


async def main(timeout: float, timings_label: str = "") -> int:
    import websockets

    from pilot.config import RUNTIME_DIR

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
                        return 1
                    ready = {}
                    request_id = 3
                    while time.monotonic() < deadline:
                        ready = await request(ws, "startup_status", request_id)
                        request_id += 1
                        if (ready.get("result") or {}).get("state") == "ready":
                            print(f"CHECK first-run/daemon-ready: PASS - after {time.monotonic() - started:.0f} s")
                            if not timings_label:
                                return 0
                            # Then wait for every late part, and report the daemon's own times.
                            status = ready.get("result") or {}
                            while time.monotonic() < deadline and "all_started" not in status.get("timings", {}):
                                await asyncio.sleep(2)
                                status = (await request(ws, "startup_status", request_id)).get("result") or {}
                                request_id += 1
                            lines, ok = timing_checks(
                                timings_label, status.get("timings") or {}, status.get("late_parts") or {}
                            )
                            for line in lines:
                                print(line)
                            return 0 if ok else 1
                        await asyncio.sleep(2)
                    print(f"CHECK first-run/daemon-ready: FAIL - still {ready} after {timeout:.0f} s")
                    return 1
            except Exception as exc:  # not listening yet
                last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(2)
    print(
        f"CHECK first-run/daemon-answers: FAIL - no answer in {timeout:.0f} s ({last_error or 'no auth token written'})"
    )
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
