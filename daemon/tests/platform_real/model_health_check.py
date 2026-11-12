"""What "No NVIDIA card" claims, seen on a real machine without one (the macOS runner, Apple Silicon, and the Linux
and Windows runners): run against a running daemon, with the environment's Python.

Asks the daemon's own ``autonomy_review_status`` RPC (what Settings shows for Gnomon, the local models and the GPU;
the window's token from the runtime folder) and checks:
- gnomon: not running and not on CUDA, and no Gnomon worker process was started (on a machine without CUDA it is
  either never asked to load, or says it is not supported here);
- every local model entry (voice, recognisers, screen reading) has a device other than CUDA;
- the GPU status reads kind 'cpu' (no NVIDIA card), no cards; on the macOS runner kind 'apple' with memory shared.
Prints each entry as the window would see it, then CHECK lines; exit code 1 if one fails.

What this cannot show: an integrated GPU, an AMD card, or a machine with a broken NVIDIA driver.
"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


async def ask_status() -> dict:
    import websockets

    sys.path.insert(0, str(HERE))
    import first_run_rpc_check as rpc

    from pilot.config import RUNTIME_DIR

    token = (RUNTIME_DIR / "auth_token").read_text(encoding="utf-8").strip()
    async with websockets.connect("ws://127.0.0.1:8785", open_timeout=10) as ws:
        await rpc.request_auth(ws, token)
        return (await rpc.request(ws, "autonomy_review_status", 2, timeout=60)).get("result") or {}


def gnomon_workers() -> list[str]:
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            line = " ".join(process.info.get("cmdline") or [])
        except Exception:
            continue
        if "pilot.intelligence.gnomon_worker" in line:
            found.append(f"{process.info['pid']}: {line[:120]}")
    return found


def main() -> int:
    status = asyncio.run(ask_status())
    print("status.gnomon:", json.dumps(status.get("gnomon"), default=str)[:600])
    for entry in status.get("local_models") or []:
        print("status.local_models:", json.dumps(entry, default=str)[:400])
    print("status.gpu:", json.dumps(status.get("gpu"), default=str)[:600])
    results = []

    gnomon = status.get("gnomon") or {}
    workers = gnomon_workers()
    ok = bool(gnomon) and gnomon.get("state") != "ready" and gnomon.get("device") != "cuda" and not workers
    results.append(
        (
            ok,
            f"CHECK no-nvidia/gnomon: {'PASS' if ok else 'FAIL'} - state={gnomon.get('state')!r}, "
            f"supported={gnomon.get('supported')!r}, reason={gnomon.get('reason')!r}, workers started: "
            f"{workers or 'none'}",
        )
    )

    entries = status.get("local_models") or []
    cuda = [entry.get("name") for entry in entries if "cuda" in str(entry.get("device") or "").lower()]
    ok = bool(entries) and not cuda
    devices = sorted({str(entry.get("device")) for entry in entries})
    results.append(
        (
            ok,
            f"CHECK no-nvidia/models-on-processor: {'PASS' if ok else 'FAIL'} - {len(entries)} entries, devices "
            f"{devices}, on CUDA: {cuda or 'none'}",
        )
    )

    machine = (status.get("gpu") or {}).get("machine") or {}
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    expect = "apple" if apple else "cpu"
    ok = machine.get("kind") == expect and (machine.get("unified_memory") is True) == apple and not machine.get("gpus")
    results.append(
        (
            ok,
            f"CHECK no-nvidia/gpu-status: {'PASS' if ok else 'FAIL'} - this machine: {platform.system()} "
            f"{platform.machine()}; Heliox reads kind={machine.get('kind')!r} (expected {expect!r}), "
            f"unified memory={machine.get('unified_memory')!r}, cards={machine.get('gpus')!r}",
        )
    )
    for _ok, line in results:
        print(line)
    return 0 if all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    sys.exit(main())
