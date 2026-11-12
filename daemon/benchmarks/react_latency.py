"""Benchmark the non-LLM ReAct latency for a simple CPU usage request.

Run from ``daemon``:
    python benchmarks/react_latency.py --iterations 10 --profile
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import platform
import pstats
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot.real_folders import use_scratch_folders  # noqa: E402 - before pilot.config is imported

BENCH_STATE_DIR = Path(__file__).resolve().parent / ".bench-state"
# Every Heliox folder (cache and setup ones too) under the bench's own; the XDG variables set only if unset left a
# person's own XDG folders, and ~/.cache/heliox always, in use.
use_scratch_folders(BENCH_STATE_DIR)


CPU_PLAN_JSON = """{
  "explanation": "Check current CPU usage.",
  "actions": [
    {
      "action_type": "cpu_usage",
      "target": "cpu",
      "parameters": {},
      "requires_root": false,
      "destructive": false,
      "reversible": true,
      "rollback_action": null,
      "use_previous_output": false
    }
  ]
}"""


class StubModelRouter:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self.generate_calls += 1
        return CPU_PLAN_JSON


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)


class NoopReasoning:
    def reset(self) -> None:
        return None


class NoopReflector:
    async def get_improvement_context(self, query: str) -> str:  # noqa: ARG002
        return ""

    async def reflect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}


class NoopMemory:
    async def get_context(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        return ""

    async def get_history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ARG002
        return []

    async def put_working(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def record(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


@dataclass
class BenchmarkHarness:
    server: Any
    model: StubModelRouter

    async def close(self) -> None:
        orchestrator = getattr(self.server, "_orchestrator", None)
        if orchestrator is not None:
            await orchestrator.stop()


async def build_harness() -> BenchmarkHarness:
    from pilot.agents.destructive_critic import DestructiveCriticAgent
    from pilot.agents.executor import Executor
    from pilot.agents.multi_agent import MultiAgentRouter
    from pilot.agents.orchestrator import AgentOrchestrator
    from pilot.agents.planner import Planner
    from pilot.agents.system_agent import SystemAgent
    from pilot.agents.verifier import Verifier
    from pilot.config import PilotConfig
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.risk_gate import get_risk_gate
    from pilot.security.validator import ActionValidator
    from pilot.server import PilotServer
    from pilot.system.sysinfo import prepare_system_probes

    config = PilotConfig()
    model = StubModelRouter()
    memory = NoopMemory()

    validator = ActionValidator(config)
    permissions = PermissionChecker(config)
    audit = AuditLogger(BENCH_STATE_DIR / "audit.jsonl")
    executor = Executor(config, validator, permissions, audit)
    verifier = Verifier(model)  # type: ignore[arg-type]

    orchestrator = AgentOrchestrator(model)  # type: ignore[arg-type]
    orchestrator.register_agent(SystemAgent(model, executor))  # type: ignore[arg-type]

    server = PilotServer(config)
    server._planner = Planner(model, memory)  # noqa: SLF001
    server._executor = executor  # noqa: SLF001
    server._verifier = verifier  # noqa: SLF001
    # Production setup imports and constructs the critic (and therefore the
    # risk/world-model stack) before the daemon reports ready. Mirroring that
    # lifecycle keeps the first measured request from including setup imports.
    server._destructive_critic = DestructiveCriticAgent(model)  # type: ignore[arg-type]  # noqa: SLF001
    get_risk_gate()
    await prepare_system_probes()
    server._permission_checker = permissions  # noqa: SLF001
    server._reflector = NoopReflector()  # noqa: SLF001
    server._multi_agent = MultiAgentRouter(model)  # type: ignore[arg-type]  # noqa: SLF001
    server._orchestrator = orchestrator  # noqa: SLF001
    server._reasoning = None  # noqa: SLF001
    server._screen_vision = None  # noqa: SLF001
    server._memory = memory  # noqa: SLF001
    return BenchmarkHarness(server=server, model=model)


async def run_once(
    harness: BenchmarkHarness,
    query: str = "What's my CPU usage?",
    expected_action: str = "cpu_usage",
) -> float:
    ws = FakeWebSocket()
    start = time.perf_counter()
    result = await harness.server._handle_execute(  # noqa: SLF001
        {"input": query, "dry_run": False},
        ws,  # type: ignore[arg-type]
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if result.get("status") != "success":
        raise RuntimeError(f"Unexpected benchmark result: {result}")
    result_actions = [
        item.get("action", {}).get("action_type") for item in result.get("results", []) if isinstance(item, dict)
    ]
    if result_actions != [expected_action]:
        raise RuntimeError(f"Expected {expected_action!r}, got actions {result_actions!r}")
    return elapsed_ms


async def benchmark(iterations: int) -> tuple[list[float], int]:
    harness = await build_harness()
    try:
        timings = []
        for _ in range(iterations):
            timings.append(await run_once(harness))
        return timings, harness.model.generate_calls
    finally:
        await harness.close()


async def benchmark_with_harness(harness: BenchmarkHarness, iterations: int) -> tuple[list[float], int]:
    timings = []
    generate_calls_before = harness.model.generate_calls
    for _ in range(iterations):
        timings.append(await run_once(harness))
    return timings, harness.model.generate_calls - generate_calls_before


async def profiled_benchmark(iterations: int) -> tuple[list[float], int]:
    harness = await build_harness()
    try:
        return await benchmark_with_harness(harness, iterations)
    finally:
        await harness.close()


def print_summary(timings: list[float], generate_calls: int) -> None:
    print(f"iterations: {len(timings)}")
    print(f"model_generate_calls: {generate_calls}")
    print(f"mean_ms: {statistics.mean(timings):.2f}")
    print(f"median_ms: {statistics.median(timings):.2f}")
    print(f"min_ms: {min(timings):.2f}")
    print(f"max_ms: {max(timings):.2f}")


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without optional dependencies."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one timing is required")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(timings: list[float]) -> dict[str, float | int]:
    """Build distribution statistics suitable for checked-in evidence."""
    if not timings:
        raise ValueError("at least one timing is required")
    return {
        "iterations": len(timings),
        "mean_ms": round(statistics.mean(timings), 3),
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(_percentile(timings, 0.95), 3),
        "p99_ms": round(_percentile(timings, 0.99), 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
        "stdev_ms": round(statistics.stdev(timings), 3) if len(timings) > 1 else 0.0,
    }


async def benchmark_report(iterations: int, warmup: int) -> dict[str, Any]:
    """Measure ready-daemon cold and steady-state latency separately."""
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    harness = await build_harness()
    try:
        cold_ms = await run_once(harness)
        for _ in range(warmup):
            await run_once(harness)
        timings, measured_model_calls = await benchmark_with_harness(harness, iterations)
        return {
            "schema_version": "2.0.0",
            "benchmark": "guarded_local_cpu_usage",
            "scope": (
                "Ready-daemon guarded CPU usage request: local planning, routing, "
                "risk assessment, execution, postcondition verification, and response shaping"
            ),
            "environment": {
                "operating_system": platform.system(),
                "operating_system_release": platform.release(),
                "python": platform.python_version(),
                "warmup_iterations": warmup,
            },
            "cold_ready_request_ms": round(cold_ms, 3),
            "steady_state": summarize(timings),
            "model_generate_calls": measured_model_calls,
            "total_model_generate_calls": harness.model.generate_calls,
            "limitations": [
                "Uses real local planner, policy, executor, verifier, and response code with in-memory transport.",
                "Excludes daemon process startup, UI rendering, cloud models, networks, browsers, voice, cameras, TTS, and neural hardware.",
                "A local reproducibility snapshot is not a universal performance guarantee.",
            ],
        }
    finally:
        await harness.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=20)
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        timings, generate_calls = asyncio.run(profiled_benchmark(args.iterations))
        profiler.disable()
        print_summary(timings, generate_calls)
        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumtime")
        stats.print_stats(args.profile_limit)
        print("\n## cProfile cumulative time")
        print(output.getvalue())
    else:
        report = asyncio.run(benchmark_report(args.iterations, args.warmup))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"cold_ready_request_ms: {report['cold_ready_request_ms']:.2f}")
            steady = report["steady_state"]
            for key, value in steady.items():
                print(f"{key}: {value}")
            print(f"model_generate_calls: {report['model_generate_calls']}")


if __name__ == "__main__":
    main()
