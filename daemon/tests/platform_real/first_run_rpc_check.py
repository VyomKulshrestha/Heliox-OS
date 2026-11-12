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


async def main(timeout: float) -> int:
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
                            return 0
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
    sys.exit(asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 180.0)))
