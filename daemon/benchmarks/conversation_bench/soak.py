"""Soak: hours of mixed spoken activity on one listener, watching the process for leaks and the voice for faults.

What can be compressed: the idle time between activities (a working day is mostly silence; here each pause is a
few seconds). What cannot: the speech itself. The listener's endpointing, Smart Turn and the follow-up windows run
on the audio's own clock, and the recognisers must keep up with it, so audio plays at real time (``--pace`` above 1
feeds it faster; the run records how far the feed fell behind, and above 1 the timings are no longer real).

Watched every ``sample_s`` seconds: resident memory of this process and of its child processes (the voice
worker), open handles (Windows) or file descriptors (macOS, Linux), threads, asyncio tasks, voice-worker starts
(restarts), and the file pump's lag. Per activity: a missed event (an addressed request that was never
dispatched) or a duplicated one (dispatched more than once). At the end: one more addressed request must be
heard and answered aloud, proving the listener is still listening and the voice still speaks.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import timeline as tl
from .scenarios import Scenario


@dataclass
class Sample:
    t: float
    rss_mb: float
    children_mb: float
    handles: int
    threads: int
    tasks: int
    worker_spawns: int
    pump_lag_s: float


@dataclass
class SoakResult:
    minutes: float
    activities: int = 0
    missed: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    still_listening: bool = False
    still_speaking: bool = False
    errors: list[str] = field(default_factory=list)


def process_sample(t: float, worker_spawns: int, pump_lag: float) -> Sample:
    """This process's resources, with psutil when present (the standard library alone gives threads only)."""
    threads = threading.active_count()
    try:
        tasks = len(asyncio.all_tasks())
    except RuntimeError:
        tasks = 0
    try:
        import psutil

        me = psutil.Process(os.getpid())
        rss = me.memory_info().rss / 2**20
        children = sum(c.memory_info().rss for c in me.children(recursive=True)) / 2**20
        handles = me.num_handles() if hasattr(me, "num_handles") else me.num_fds()
        threads = me.num_threads()
    except Exception:
        rss, children, handles = 0.0, 0.0, -1
    return Sample(round(t, 1), round(rss, 1), round(children, 1), handles, threads, tasks, worker_spawns, pump_lag)


def slope_per_hour(samples: list[Sample], attr: str) -> float:
    """Least-squares growth of ``attr`` per hour (pure), ignoring the first tenth (models loading)."""
    usable = samples[len(samples) // 10 :] if len(samples) >= 10 else samples
    if len(usable) < 2:
        return 0.0
    xs = [s.t / 3600 for s in usable]
    ys = [float(getattr(s, attr)) for s in usable]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    return 0.0 if var == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var


def judge(events: list[tl.Event], scenario: Scenario) -> tuple[bool, bool]:
    """(missed, duplicated) for one activity (pure)."""
    says = sum(1 for s in scenario.steps if s.say)
    dispatched = len(tl.all_of(events, tl.DISPATCH))
    addressed = bool(scenario.expect.get("woken")) and scenario.set != "not_addressed"
    # Not-addressed speech that is acted on is scored by line 2.2, not counted here as a duplicate.
    return (addressed and dispatched == 0), (scenario.set != "not_addressed" and dispatched > says)


def plan(pool: list[Scenario], seed: int = 3) -> list[Scenario]:
    """An endless-looking, reproducible mix: requests, barge-ins, control, overheard speech and meetings."""
    rng = random.Random(seed)
    weights = {"understanding": 4, "control": 3, "turn_taking": 2, "not_addressed": 2}
    usable = [s for s in pool if s.set in weights and not s.pending]
    order: list[Scenario] = []
    for _ in range(4000):
        sets = [k for k in weights if any(s.set == k for s in usable)]
        chosen = rng.choices(sets, [weights[k] for k in sets])[0]
        order.append(rng.choice([s for s in usable if s.set == chosen]))
    return order


async def run(harness: Any, pool: list[Scenario], minutes: float, sample_s: float = 10.0) -> SoakResult:
    result = SoakResult(minutes)
    start = time.perf_counter()
    stop = asyncio.Event()

    async def watch() -> None:
        while not stop.is_set():
            pump = harness.pump
            result.samples.append(
                process_sample(time.perf_counter() - start, harness.worker_spawns, pump.max_lag if pump else 0.0)
            )
            try:
                await asyncio.wait_for(stop.wait(), sample_s)
            except asyncio.TimeoutError:
                pass

    watcher = asyncio.ensure_future(watch())
    end = start + minutes * 60
    final = next((s for s in pool if s.id.startswith("tt-syn-reply-local")), None)
    try:
        for scenario in plan(pool):
            if time.perf_counter() >= end - 30:
                break
            try:
                events = await harness.run(scenario, "as-recorded")
            except Exception as error:  # a fault in one activity is recorded, the soak goes on
                result.errors.append(f"{scenario.id}: {type(error).__name__}: {error}"[:200])
                continue
            result.activities += 1
            missed, duplicated = judge(events, scenario)
            if missed:
                result.missed.append(scenario.id)
            if duplicated:
                result.duplicated.append(scenario.id)
            await asyncio.sleep(2.0)  # the compressed idle time between activities
        if final is not None:
            events = await harness.run(final, "as-recorded")
            result.still_listening = bool(tl.all_of(events, tl.DISPATCH))
            result.still_speaking = bool([e for e in tl.all_of(events, tl.AUDIO_STARTS) if not e.data.get("setup")])
    finally:
        stop.set()
        await watcher
        result.samples.append(
            process_sample(
                time.perf_counter() - start, harness.worker_spawns, harness.pump.max_lag if harness.pump else 0
            )
        )
    return result


def summary(result: SoakResult) -> str:
    s = result.samples
    if not s:
        return "no samples"
    first, last = s[min(len(s) - 1, max(1, len(s) // 10))], s[-1]
    lines = [
        f"soak: {result.minutes:.0f} min, {result.activities} activities",
        f"memory (this process): {first.rss_mb:.0f} -> {last.rss_mb:.0f} MB after warm-up, max "
        f"{max(x.rss_mb for x in s):.0f} MB, trend {slope_per_hour(s, 'rss_mb'):+.0f} MB/hour",
        f"memory (voice worker and other children): {first.children_mb:.0f} -> {last.children_mb:.0f} MB",
        f"handles/fds: {first.handles} -> {last.handles} (trend {slope_per_hour(s, 'handles'):+.0f}/hour); "
        f"threads: {first.threads} -> {last.threads}; asyncio tasks: {first.tasks} -> {last.tasks}",
        f"voice worker starts: {last.worker_spawns} (1 is the first load; more are restarts)",
        f"audio feed fell behind real time by at most {last.pump_lag_s:.2f} s",
        f"missed: {len(result.missed)} {sorted(set(result.missed))[:5]}; duplicated: {len(result.duplicated)} "
        f"{sorted(set(result.duplicated))[:5]}; errors: {len(result.errors)} {result.errors[:3]}",
        f"at the end: still listening {result.still_listening}, still speaking {result.still_speaking}",
    ]
    return "\n".join(lines)
