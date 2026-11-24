"""Pocket TTS's speed on this machine: the first audible chunk and the real-time factor, through Heliox's own stream
(pilot.system.pocket_stream: numbers as words, the quiet lead dropped), for 1, 2 and 4 threads. Report only.

The question it answers: does Pocket stream without gaps on a standard 4-core runner? Streaming needs the real-time
factor (seconds of making per second of audio) under 1.0, with room to spare; on one development laptop (16 cores)
it was 0.32 to 0.35 on one thread, with the first audible chunk at a median 0.13 s. On GitHub's 4-core Linux and
Windows runners every piece streamed without gaps, and one thread was fastest (0.31 to 0.39 against 0.42 to 0.49 on
four), so the voice worker gives Pocket one thread (local_tts_worker.ENGINE_THREADS).

    python tests/platform_real/pocket_speed_report.py [--threads 1,2,4] [--summary FILE]

Prints the processor's model and architecture, one POCKET line per thread count and, with --summary, a Markdown
table (GitHub's step summary). The model is fetched by pocket-tts itself on first use unless HF_HUB_OFFLINE=1 and it
is cached. Every Heliox folder is a scratch folder (use_scratch_folders). Nothing is played.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

PIECES = [
    "Done.", "Okay.", "It's 10:42.", "Volume is at 30.", "Opening Notepad.",
    "Siobhan Nakamura called at 3:15.", "The rate went up to 3.5 percent.", "That was in March 2026.",
    "The nineteenth is a Thursday.", "You have three things on your list today.",
    "Okay, I'll remind you to call the bank at four.", "Tuesday the seventeenth, or Thursday the nineteenth?",
    "I moved the meeting with Priya to Friday at ten.",
    "Your next call starts in twelve minutes, with the design team.",
    "Siobhan Nakamura sent the budget at 10:42, and it is 3.5 percent over.",
    "The battery is at 64 percent and charging, so you have about three hours left.",
]  # fmt: skip


def summarise(rows: list[dict]) -> dict:
    """Medians and worsts of one thread count's rows ({"first_s", "made_s", "audio_s"})."""
    firsts = [row["first_s"] for row in rows]
    rtfs = [row["made_s"] / row["audio_s"] for row in rows if row["audio_s"] > 0]
    return {
        "pieces": len(rows),
        "first_median_s": round(statistics.median(firsts), 3) if firsts else None,
        "first_worst_s": round(max(firsts), 3) if firsts else None,
        "rtf_median": round(statistics.median(rtfs), 3) if rtfs else None,
        "rtf_worst": round(max(rtfs), 3) if rtfs else None,
        "streams_without_gaps": bool(rtfs) and max(rtfs) < 1.0,
    }


def name_from_cpuinfo(text: str) -> str:
    """The processor's model from Linux's /proc/cpuinfo: "model name" (x86), else "Hardware" or "Processor" (Arm)."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().casefold(), value.strip()
        if value and key not in found:
            found[key] = value
    for key in ("model name", "hardware", "processor", "cpu model"):
        value = found.get(key, "")
        if value and not value.isdigit():  # x86's "processor" line is the core's number
            return value
    return ""


def _windows_processor_name() -> str:
    import winreg  # type: ignore[import-not-found]

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()


def _macos_processor_name() -> str:
    answer = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
    return answer.stdout.strip()


def _linux_processor_name() -> str:
    return name_from_cpuinfo(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"))


def processor_facts(
    system: str | None = None,
    machine: str | None = None,
    readers: dict[str, Callable[[], str]] | None = None,
    fallback: Callable[[], str] = platform.processor,
) -> dict[str, str]:
    """{"processor": its model, "architecture": x86_64, AMD64, arm64, ...}: the speed depends on both, and a 4-core
    runner of one system is not the same processor as another's."""
    system = system or platform.system()
    readers = readers if readers is not None else {
        "Windows": _windows_processor_name, "Darwin": _macos_processor_name, "Linux": _linux_processor_name,
    }  # fmt: skip
    name = ""
    reader = readers.get(system)
    if reader is not None:
        try:
            name = reader()
        except Exception:
            name = ""
    if not name:
        try:
            name = fallback() or ""
        except Exception:
            name = ""
    return {
        "processor": " ".join(name.split()) or "unknown",
        "architecture": machine or platform.machine() or "unknown",
    }


def heading(machine: dict) -> str:
    """The summary's first line: the processor, its architecture, cores and memory, load and resident memory."""
    return (
        f"### Pocket TTS on {machine['processor']} ({machine['architecture']}), {machine['cores']} cores, "
        f"{machine['memory_gb']} GB (load {machine['load_s']} s, {machine['rss_mb']} MB resident)"
    )


def measure(engine, threads: int) -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    engine.prepare()
    rows = []
    for text in PIECES:
        first = None
        samples = 0
        started = time.perf_counter()
        for chunk in engine.stream(text):
            if first is None:
                first = time.perf_counter() - started
            samples += len(np.asarray(chunk))
        made = time.perf_counter() - started
        rows.append({"first_s": first or made, "made_s": made, "audio_s": samples / engine.sample_rate})
    return {"threads": threads, **summarise(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pilot.real_folders import use_scratch_folders

    use_scratch_folders(Path(os.environ["HELIOX_BENCH_ROOT"]) if os.environ.get("HELIOX_BENCH_ROOT") else None)
    import psutil

    from pilot.system.pocket_stream import PocketStream

    started = time.perf_counter()
    engine = PocketStream()
    engine.load()
    load_s = round(time.perf_counter() - started, 2)
    machine = {
        **processor_facts(),
        "cores": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "load_s": load_s,
        "rss_mb": psutil.Process().memory_info().rss >> 20,
    }
    print("POCKET machine " + json.dumps(machine), flush=True)
    results = []
    for threads in (int(part) for part in args.threads.split(",") if part.strip()):
        result = measure(engine, threads)
        results.append(result)
        print("POCKET " + json.dumps(result), flush=True)
    if args.summary:
        lines = [
            heading(machine),
            "",
            "| threads | first audible chunk, median / worst | real-time factor, median / worst | streams without gaps |",
            "|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['threads']} | {r['first_median_s']} / {r['first_worst_s']} s | {r['rtf_median']} / "
                f"{r['rtf_worst']} | {'yes' if r['streams_without_gaps'] else 'no'} |"
            )
        with open(args.summary, "a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
