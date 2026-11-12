"""The update path when port 8785 is held by a program that is not Heliox (CI runners: Windows, macOS, Linux).

A plain Python process listens on the port. update.py's own check (make_way, with the real stop helper and the real
netstat/tasklist, lsof or ss) must refuse to go ahead, name that process and its id, say how to stop it on this
system, and leave it running. Run with the environment's Python. Prints CHECK lines; exit code 1 if one fails.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = 8785
HOLDER = (
    f"import socket, time; s = socket.socket(); s.bind(('127.0.0.1', {PORT})); s.listen(); print('ready', flush=True); "
    "time.sleep(600)"
)


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "heliox_update", ROOT / "tauri-app" / "src-tauri" / "setup" / "update.py"
    )
    update = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(update)

    holder = subprocess.Popen([sys.executable, "-c", HOLDER], stdout=subprocess.PIPE, text=True)
    try:
        if (holder.stdout.readline() if holder.stdout else "").strip() != "ready":
            print("CHECK foreign-port/named: FAIL - the stand-in program could not hold the port")
            return 1
        started = time.monotonic()
        go_ahead, hint = update.make_way(Path(sys.executable), port=PORT)
        took = time.monotonic() - started
        # On Windows a virtual environment's python.exe is a launcher; the listening interpreter is its child.
        import psutil

        pids = {holder.pid, *(child.pid for child in psutil.Process(holder.pid).children(recursive=True))}
        named = any(f"process {pid}" in hint for pid in pids) and "python" in hint.lower()
        how = {"win32": "Task Manager", "darwin": "Activity Monitor"}.get(sys.platform, "kill")
        ok = not go_ahead and named and how in hint and holder.poll() is None
        print(
            f"CHECK foreign-port/named: {'PASS' if ok else 'FAIL'} - the update {'stopped' if not go_ahead else 'went ahead'} "
            f"in {took:.1f} s and said: {hint!r}; the program holding the port is "
            f"{'still running' if holder.poll() is None else 'gone'}"
        )
        return 0 if ok else 1
    finally:
        holder.terminate()
        holder.wait(30)


if __name__ == "__main__":
    sys.exit(main())
