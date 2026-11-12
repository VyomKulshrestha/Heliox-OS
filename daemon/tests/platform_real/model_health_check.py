"""What "No NVIDIA card" claims, seen on a real machine without one (the macOS runner, Apple Silicon, and the Linux
and Windows runners): run against the daemon the first-run job started, with the environment's Python.

Asks the daemon's own ``health`` RPC (the window's token from the runtime folder) and checks:
- gnomon: reported as not supported here, with a reason, and no Gnomon worker process was started to find out;
- every local model entry (voice, recognisers, screen reading) names the processor or the system's own engine, never
  CUDA;
- the GPU status reads kind 'cpu' (no NVIDIA card); on the macOS runner kind 'apple' with memory shared.
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


async def ask_health() -> dict:
    import websockets

    sys.path.insert(0, str(HERE))
    import first_run_rpc_check as rpc

    from pilot.config import RUNTIME_DIR

    token = (RUNTIME_DIR / "auth_token").read_text(encoding="utf-8").strip()
    async with websockets.connect("ws://127.0.0.1:8785", open_timeout=10) as ws:
        await rpc.request_auth(ws, token)
        return (await rpc.request(ws, "health", 2, timeout=60)).get("result") or {}


def gnomon_workers() -> list[str]:
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        line = " ".join(process.info.get("cmdline") or [])
        if "pilot.intelligence.gnomon_worker" in line:
            found.append(f"{process.info['pid']}: {line[:120]}")
    return found


def main() -> int:
    health = asyncio.run(ask_health())
    print("health.gnomon:", json.dumps(health.get("gnomon"), default=str)[:600])
    for entry in health.get("local_models") or []:
        print("health.local_models:", json.dumps(entry, default=str)[:400])
    print("health.gpu:", json.dumps(health.get("gpu"), default=str)[:600])
    results = []

    gnomon = health.get("gnomon") or {}
    workers = gnomon_workers()
    ok = gnomon.get("supported") is False and not workers
    results.append(
        (
            ok,
            f"CHECK no-nvidia/gnomon: {'PASS' if ok else 'FAIL'} - supported={gnomon.get('supported')!r}, "
            f"state={gnomon.get('state')!r}, workers started: {workers or 'none'}",
        )
    )

    entries = health.get("local_models") or []
    cuda = [entry.get("name") or entry.get("id") for entry in entries if "cuda" in json.dumps(entry).lower()]
    ok = bool(entries) and not cuda
    results.append(
        (
            ok,
            f"CHECK no-nvidia/models-on-processor: {'PASS' if ok else 'FAIL'} - {len(entries)} entries, "
            f"naming CUDA: {cuda or 'none'}",
        )
    )

    machine = (health.get("gpu") or {}).get("machine") or {}
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
