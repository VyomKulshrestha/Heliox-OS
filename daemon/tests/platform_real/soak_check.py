"""The weekly soak: a real daemon answering requests for hours, sampled for leaks (CI runners: Windows, macOS, Linux).

Run with the environment's Python and a fresh HOME (USERPROFILE, LOCALAPPDATA on Windows):

    python daemon/tests/platform_real/soak_check.py [minutes]      # default 120, or HELIOX_SOAK_MINUTES

The daemon's model is a local address that refuses at once, so nothing costs money and nothing waits on a network.
Requests are those a person makes that Heliox answers itself: a note taken and deleted again, the notes read, a
to-do added and removed, plus the window's own status calls (ping, health, startup status, storage sizes, the review
status). Every few minutes the daemon process is sampled: memory (RSS), threads, open files (descriptors on Linux and
macOS, handles on Windows), child processes and the event loop's task count (``health``). The first sample is taken
after a warm-up (when every late part has started); the checks compare the last third of the run with that.

Bounds (a slow leak shows as steady growth; these allow for caches filling once):
- memory: grows less than 25 % and less than 150 MB from the warm sample;
- event-loop tasks: at most 25 more; threads: at most 10 more; open files or handles: at most 50 more;
- no request fails or goes unanswered, and the slowest answer is printed.

What this cannot show: weeks of uptime, a person's real mix (voice, screen reading and models are not running on a
runner), sleep and wake, and data folders the size of a year of use (store_growth_check.py measures that growth).
Prints the samples, then CHECK lines; exit code 1 if one fails.
Off a CI runner it changes nothing and exits with 3 (ci_only.py).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("writes Heliox's config.toml in the home folder and runs a real daemon there for hours")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_daemon  # noqa: E402

WARM_UP = 120.0  # seconds after ready before the first sample: the late parts start in about 20 s
SAMPLE_EVERY = 300.0
MEMORY_GROWTH = (0.25, 150.0)  # fraction, MB
TASK_GROWTH, THREAD_GROWTH, FILE_GROWTH = 25, 10, 50


def sample(pid: int, tasks: int) -> dict:
    import psutil

    process = psutil.Process(pid)
    with process.oneshot():
        files = process.num_handles() if sys.platform == "win32" else process.num_fds()
        return {
            "rss_mb": process.memory_info().rss / 2**20,
            "threads": process.num_threads(),
            "files": files,
            "children": len(process.children(recursive=True)),
            "tasks": tasks,
        }


async def one_round(session, number: int, answers: list[float], failures: list[str]) -> None:
    async def timed_ask(text: str, expect: str) -> None:
        began = time.monotonic()
        said = await session.ask(text, timeout=120)
        answers.append(time.monotonic() - began)
        if expect not in said:
            failures.append(f"{text!r} -> {said[:160]!r}")

    async def timed_call(method: str, params: dict | None = None) -> dict:
        reply, took = await session.timed(method, params, timeout=120)
        answers.append(took)
        if "rpc_error" in reply or reply.get("status") == "error":
            failures.append(f"{method} -> {str(reply)[:160]}")
        return reply

    await timed_ask(f"note that soak round {number}", "Noted")
    await timed_ask("read my notes", "notes")
    await timed_ask("delete my last note", "Deleted")
    added = await timed_call("todo_add", {"text": f"soak to-do {number}"})
    for todo in added.get("todos") or []:
        if str(todo.get("text", "")).startswith("soak to-do") and todo.get("id"):
            await timed_call("todo_delete", {"id": todo["id"]})
    await timed_call("ping")
    await timed_call("startup_status")
    if number % 10 == 0:
        await timed_call("storage_sizes")
        await timed_call("autonomy_review_status")


async def run(minutes: float) -> int:
    real_daemon.write_config()
    daemon = real_daemon.start()
    session = None
    samples: list[tuple[float, dict]] = []
    answers: list[float] = []
    failures: list[str] = []
    try:
        session = await real_daemon.connect(240)
        ready, status = await session.wait_ready(240)
        if not ready:
            print(f"CHECK soak/ready: FAIL - {status}")
            return 1
        began = time.monotonic()
        end = began + minutes * 60
        next_sample = began + WARM_UP
        number = 0
        while time.monotonic() < end:
            if daemon.poll() is not None:
                failures.append(f"the daemon exited with code {daemon.returncode}")
                break
            await one_round(session, number, answers, failures)
            number += 1
            if time.monotonic() >= next_sample:
                health = await session.call("health", timeout=60)
                found = sample(daemon.pid, int(health.get("event_loop_tasks", -1)))
                samples.append((time.monotonic() - began, found))
                print(
                    f"SAMPLE {(time.monotonic() - began) / 60:6.1f} min, {number} rounds: rss {found['rss_mb']:.0f} MB, "
                    f"threads {found['threads']}, files {found['files']}, children {found['children']}, "
                    f"tasks {found['tasks']}",
                    flush=True,
                )
                next_sample += SAMPLE_EVERY
            await asyncio.sleep(1)
        rounds = number
    finally:
        await real_daemon.stop(daemon, session)

    lines: list[tuple[bool, str]] = []
    answers.sort()
    slowest = answers[-1] if answers else None
    p95 = answers[int(len(answers) * 0.95)] if answers else None
    ok = not failures and bool(answers)
    lines.append(
        (
            ok,
            f"CHECK soak/answers: {'PASS' if ok else 'FAIL'} - {rounds} rounds, {len(answers)} requests in "
            f"{minutes:g} min; 95th percentile {p95 if p95 is None else round(p95, 2)} s, slowest "
            f"{slowest if slowest is None else round(slowest, 2)} s; failures: {failures[:5] or 'none'}"
            f"{f' (+{len(failures) - 5} more)' if len(failures) > 5 else ''}",
        )
    )
    if len(samples) < 3:
        lines.append((False, f"CHECK soak/sampled: FAIL - only {len(samples)} samples; run longer"))
    else:
        warm = samples[0][1]
        last_third = [found for at, found in samples if at >= samples[-1][0] * 2 / 3]
        late = {key: max(found[key] for found in last_third) for key in warm}
        growth = late["rss_mb"] - warm["rss_mb"]
        ok = growth < MEMORY_GROWTH[1] and growth < warm["rss_mb"] * MEMORY_GROWTH[0]
        lines.append(
            (
                ok,
                f"CHECK soak/memory: {'PASS' if ok else 'FAIL'} - {warm['rss_mb']:.0f} MB after warm-up, at most "
                f"{late['rss_mb']:.0f} MB in the last third (+{growth:.0f} MB; bound +{MEMORY_GROWTH[0]:.0%} and "
                f"+{MEMORY_GROWTH[1]:g} MB)",
            )
        )
        for key, bound, words in (
            ("tasks", TASK_GROWTH, "event-loop tasks"),
            ("threads", THREAD_GROWTH, "threads"),
            ("files", FILE_GROWTH, "open files" if sys.platform != "win32" else "open handles"),
            ("children", 2, "child processes"),
        ):
            ok = late[key] - warm[key] <= bound
            lines.append(
                (
                    ok,
                    f"CHECK soak/{key}: {'PASS' if ok else 'FAIL'} - {words}: {warm[key]} after warm-up, at most "
                    f"{late[key]} in the last third (bound +{bound})",
                )
            )
    for _ok, line in lines:
        print(line)
    return 0 if all(ok for ok, _line in lines) else 1


if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else float(os.environ.get("HELIOX_SOAK_MINUTES", "120"))
    sys.exit(asyncio.run(run(minutes)))
