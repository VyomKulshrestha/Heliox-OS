"""Benchmark the non-LLM ReAct latency for a simple CPU usage request.

Run from ``daemon``:
    python benchmarks/react_latency.py --iterations 10 --profile
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import os
import pstats
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCH_STATE_DIR = Path(__file__).resolve().parent / ".bench-state"
BENCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str(BENCH_STATE_DIR / "config"))
os.environ.setdefault("XDG_DATA_HOME", str(BENCH_STATE_DIR / "data"))
os.environ.setdefault("XDG_STATE_HOME", str(BENCH_STATE_DIR / "state"))
os.environ.setdefault("XDG_RUNTIME_DIR", str(BENCH_STATE_DIR / "runtime"))


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
    async def record(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


@dataclass
class BenchmarkHarness:
    server: Any
    model: StubModelRouter


async def build_harness() -> BenchmarkHarness:
    from pilot.agents.executor import Executor
    from pilot.agents.multi_agent import MultiAgentRouter
    from pilot.agents.orchestrator import AgentOrchestrator
    from pilot.agents.planner import Planner
    from pilot.agents.system_agent import SystemAgent
    from pilot.agents.verifier import Verifier
    from pilot.config import PilotConfig
    from pilot.memory.store import MemoryStore
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator
    from pilot.server import PilotServer

    config = PilotConfig()
    model = StubModelRouter()
    memory = MemoryStore()
    await memory.initialize()

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
    server._reflector = NoopReflector()  # noqa: SLF001
    server._multi_agent = MultiAgentRouter(model)  # type: ignore[arg-type]  # noqa: SLF001
    server._orchestrator = orchestrator  # noqa: SLF001
    server._reasoning = None  # noqa: SLF001
    server._screen_vision = None  # noqa: SLF001
    server._memory = NoopMemory()  # noqa: SLF001
    return BenchmarkHarness(server=server, model=model)


async def run_once(harness: BenchmarkHarness) -> float:
    ws = FakeWebSocket()
    start = time.perf_counter()
    result = await harness.server._handle_execute(  # noqa: SLF001
        {"input": "What's my CPU usage?", "dry_run": False},
        ws,  # type: ignore[arg-type]
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if result.get("status") != "success":
        raise RuntimeError(f"Unexpected benchmark result: {result}")
    return elapsed_ms


async def benchmark(iterations: int) -> tuple[list[float], int]:
    harness = await build_harness()
    timings = []
    for _ in range(iterations):
        timings.append(await run_once(harness))
    return timings, harness.model.generate_calls


async def benchmark_with_harness(harness: BenchmarkHarness, iterations: int) -> tuple[list[float], int]:
    timings = []
    generate_calls_before = harness.model.generate_calls
    for _ in range(iterations):
        timings.append(await run_once(harness))
    return timings, harness.model.generate_calls - generate_calls_before


def print_summary(timings: list[float], generate_calls: int) -> None:
    print(f"iterations: {len(timings)}")
    print(f"model_generate_calls: {generate_calls}")
    print(f"mean_ms: {statistics.mean(timings):.2f}")
    print(f"median_ms: {statistics.median(timings):.2f}")
    print(f"min_ms: {min(timings):.2f}")
    print(f"max_ms: {max(timings):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=20)
    args = parser.parse_args()

    if args.profile:
        harness = asyncio.run(build_harness())
        profiler = cProfile.Profile()
        profiler.enable()
        timings, generate_calls = asyncio.run(benchmark_with_harness(harness, args.iterations))
        profiler.disable()
        print_summary(timings, generate_calls)
        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumtime")
        stats.print_stats(args.profile_limit)
        print("\n## cProfile cumulative time")
        print(output.getvalue())
    else:
        timings, generate_calls = asyncio.run(benchmark(args.iterations))
        print_summary(timings, generate_calls)


if __name__ == "__main__":
    main()
