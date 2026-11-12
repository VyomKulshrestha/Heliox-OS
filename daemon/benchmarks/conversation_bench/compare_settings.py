"""Comparing settings inside one timing job: one runner, the settings interleaved, the differences with their spread.

Two runs of the same commit on shared runners differed by up to 0.6 s in a stage (runs 43267447050 and 43271354855,
12 November), so one run per setting cannot tell a 0.1 to 0.3 s change. Here every scenario runs once per setting,
back to back, in an order that rotates each repeat so a runner that drifts hits every setting alike; each stage's
difference against the first setting is taken within each repeat, and reported as the median with its range over
the repeats. A calibration line (the CPU's name, a fixed pure-Python loop, one Parakeet pass on a fixed clip) lets
runs on different runners be put on one scale.

Settings are given as "name=v1,v2,..." (``--settings parakeet_threads=2,3,4``); the names are in SETTINGS.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import stages


def _parakeet_threads(harness: Any, value: str) -> None:
    """Parakeet's thread count for the listener's final pass, loaded again (outside any scenario's timing)."""
    listener = harness.listener
    transcriber = listener._parakeet() if listener is not None else None
    if transcriber is None:
        raise RuntimeError("parakeet_threads: the listener has no Parakeet final pass")
    threads = int(value)
    if transcriber.threads != threads or transcriber._recognizer is None:
        transcriber.threads = threads
        transcriber._recognizer = None
        transcriber.load()


SETTINGS: dict[str, Callable[[Any, str], None]] = {"parakeet_threads": _parakeet_threads}


def parse(spec: str) -> tuple[str, list[str]]:
    """ "parakeet_threads=2,3,4" -> ("parakeet_threads", ["2", "3", "4"]) (pure). "" -> ("", [""])."""
    if not spec.strip():
        return "", [""]
    name, _, values = spec.partition("=")
    name = name.strip()
    if name not in SETTINGS:
        raise ValueError(f"unknown setting {name!r}; known: {', '.join(SETTINGS)}")
    chosen = [v.strip() for v in values.split(",") if v.strip()]
    if len(chosen) < 1:
        raise ValueError(f"no values for {name}")
    return name, chosen


def order(values: list[str], repeat: int) -> list[str]:
    """The settings' order for one repeat, rotated by one each time so each comes first as often (pure)."""
    shift = repeat % len(values)
    return values[shift:] + values[:shift]


def differences(per: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, dict[str, dict[str, float]]]:
    """For each setting after the first and each stage: its median minus the first setting's, within each repeat,
    then the median and range of those differences over the repeats (pure).

    ``per[setting][repeat]`` is that repeat's list of stage breakdowns (stages.breakdown) for one scenario group."""
    settings = list(per)
    if len(settings) < 2:
        return {}
    base = settings[0]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in settings[1:]:
        by_stage: dict[str, list[float]] = {}
        for repeat in per[setting]:
            if repeat not in per[base]:
                continue
            mine = stages.summarise(per[setting][repeat])
            theirs = stages.summarise(per[base][repeat])
            for stage in mine.keys() & theirs.keys():
                by_stage.setdefault(stage, []).append(float(mine[stage]["wall"]) - float(theirs[stage]["wall"]))
        out[setting] = {
            stage: {
                "median": round(statistics.median(diffs), 3),
                "low": round(min(diffs), 3),
                "high": round(max(diffs), 3),
                "repeats": len(diffs),
            }
            for stage, diffs in sorted(by_stage.items())
        }
    return out


def difference_table(name: str, groups: dict[str, dict[str, dict[int, list[dict[str, Any]]]]]) -> str:
    """Markdown: per scenario group and stage, each setting's median, and its difference from the first setting with
    the range over repeats (pure)."""
    lines = [
        f"| scenarios | stage | {name} | median, wall clock | difference from {name}={{first}}, median (range) | repeats |",
        "|---|---|---|---|---|---|",
    ]
    for group, per in groups.items():
        settings = list(per)
        if not settings:
            continue
        lines[0] = lines[0].replace("{first}", settings[0])
        diffs = differences(per)
        for setting in settings:
            pooled = [b for runs in per[setting].values() for b in runs]
            for stage, v in stages.summarise(pooled).items():
                d = diffs.get(setting, {}).get(stage)
                shown = "-" if d is None else f"{d['median']:+.2f} s ({d['low']:+.2f} to {d['high']:+.2f})"
                count = "-" if d is None else str(d["repeats"])
                lines.append(f"| {group} | {stage} | {setting} | {v['wall']:.2f} s | {shown} | {count} |")
    return "\n".join(lines)


def cpu_name() -> str:
    """The processor's name as the system gives it ("" when it does not)."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ""


def python_loop_seconds(iterations: int = 3_000_000) -> float:
    """A fixed pure-Python loop, the best of three: how fast this runner's single core is."""
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        total = 0
        for i in range(iterations):
            total += i & 7
        best = min(best, time.perf_counter() - started)
    return round(best, 3)


def parakeet_pass_seconds(transcriber: Any, clip: Any) -> float | None:
    """One Parakeet pass on a fixed clip after a warm one, the best of three; None without Parakeet."""
    if transcriber is None or not transcriber.load():
        return None
    transcriber.transcribe(clip)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        transcriber.transcribe(clip)
        best = min(best, time.perf_counter() - started)
    return round(best, 3)


def calibration_line(cpu: str, cores: int | None, loop_s: float, pass_s: float | None, threads: int | None) -> str:
    """CALIBRATION line printed beside the scorecard (pure)."""
    parakeet = "-" if pass_s is None else f"{pass_s:.3f} s on {threads} threads"
    return f"CALIBRATION cpu={cpu or 'unknown'!r} cores={cores} python_loop={loop_s:.3f} s parakeet_pass={parakeet}"
