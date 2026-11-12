"""stop() timed on a real daemon with a request in flight (CI runners: Windows, macOS, Linux).

Run with the environment's Python and HOME (USERPROFILE, LOCALAPPDATA on Windows) set to a fresh, empty folder: the
script writes its own config there. The daemon's model is pointed at a local port that accepts and never answers, a
request is sent, and once the daemon is seen calling that port it is asked to stop (``daemon_stop``, the same clean
stop as SIGTERM, on every system). The stated bounds are 10 s per step and 30 s for the whole stop; the check times
the stop from the request to the process exit and lists every step the daemon's log says it left behind.

What this cannot show: a stop with the voice worker or Gnomon's worker running (neither starts on a runner without
models or an NVIDIA card). Prints CHECK lines; exit code 1 if one fails.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOTAL_BOUND = 30.0
MARGIN = 5.0


def stall_server() -> tuple[int, list[int]]:
    """A local port that accepts and never answers; the list counts connections."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    seen: list[int] = []
    held: list[socket.socket] = []

    def run() -> None:
        while True:
            connection, _address = listener.accept()
            held.append(connection)  # kept open, never answered
            seen.append(1)

    threading.Thread(target=run, daemon=True).start()
    return listener.getsockname()[1], seen


async def send(method: str, params: dict, *, wait: bool) -> dict | None:
    import websockets

    sys.path.insert(0, str(HERE))
    import first_run_rpc_check as rpc

    from pilot.config import RUNTIME_DIR

    token = (RUNTIME_DIR / "auth_token").read_text(encoding="utf-8").strip()
    ws = await websockets.connect("ws://127.0.0.1:8785", open_timeout=10)
    await rpc.request_auth(ws, token)
    await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 9}))
    if not wait:
        return {"ws": ws}
    try:
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if message.get("id") == 9:
                return message
    finally:
        await ws.close()


async def run() -> int:
    sys.path.insert(0, str(HERE))
    import first_run_rpc_check as rpc

    from pilot.config import CONFIG_FILE, LOG_FILE

    port, seen = stall_server()
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        f'first_run_complete = true\n\n[model]\nprovider = "ollama"\nollama_base_url = "http://127.0.0.1:{port}"\n',
        encoding="utf-8",
    )
    daemon = subprocess.Popen(
        [sys.executable, "-m", "pilot.server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        if await rpc.main(240) != 0:
            print("CHECK stop/in-flight: FAIL - the daemon did not become ready")
            return 1
        in_flight = await send("execute", {"input": "write a short poem about rain"}, wait=False)
        deadline = time.monotonic() + 60
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        if not seen:
            print("CHECK stop/in-flight: FAIL - the daemon never called the model port, so nothing was in flight")
            return 1
        print(f"INFO a model call is in flight ({len(seen)} connection(s) to the silent port)")
        started = time.monotonic()
        await send("daemon_stop", {"reason": "stop timing check"}, wait=True)
        while daemon.poll() is None and time.monotonic() - started < TOTAL_BOUND * 3:
            await asyncio.sleep(0.1)
        took = time.monotonic() - started
        try:
            await in_flight["ws"].close()
        except Exception:
            pass
        left = [
            line for line in LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines() if "Stopping: " in line
        ]
        stopped = daemon.poll() is not None
        ok = stopped and took <= TOTAL_BOUND + MARGIN
        print(
            f"CHECK stop/in-flight: {'PASS' if ok else 'FAIL'} - with a model call in flight the daemon "
            f"{'exited' if stopped else 'was still running'} {took:.1f} s after it was asked to stop "
            f"(stated bound {TOTAL_BOUND:g} s, 10 s per step)"
        )
        for line in left:
            print(f"INFO {line.strip()[-200:]}")
        return 0 if ok else 1
    finally:
        if daemon.poll() is None:
            daemon.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
