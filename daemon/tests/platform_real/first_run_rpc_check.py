"""After a first-run setup from nothing: does the daemon that was just installed answer its RPC?

Run with the new environment's Python while that daemon starts. Waits for the auth token the daemon writes, connects
to ws://127.0.0.1:8785, authenticates, calls ``ping``, and prints one CHECK line. Exit code 0 when it answered.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


async def main(timeout: float) -> int:
    import websockets

    from pilot.config import RUNTIME_DIR

    token_file = RUNTIME_DIR / "auth_token"
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if token_file.exists():
            try:
                async with websockets.connect("ws://127.0.0.1:8785", open_timeout=5) as ws:
                    token = token_file.read_text(encoding="utf-8").strip()
                    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": token}, "id": 1}))
                    auth = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 2}))
                    reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    ok = "error" not in auth and "error" not in reply
                    took = timeout - (deadline - time.monotonic())
                    print(f"CHECK first-run/daemon-answers: {'PASS' if ok else 'FAIL'} - {reply} after {took:.0f} s")
                    return 0 if ok else 1
            except Exception as exc:  # not listening yet
                last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(2)
    print(
        f"CHECK first-run/daemon-answers: FAIL - no answer in {timeout:.0f} s ({last_error or 'no auth token written'})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 180.0)))
