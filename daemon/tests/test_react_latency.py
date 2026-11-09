"""Regression tests for simple ReAct loop latency optimizations."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.local_status_suite import benchmark_suite
from benchmarks.react_latency import benchmark, benchmark_report, summarize
from pilot.actions import ActionType
from pilot.agents.planner import Planner
from pilot.memory.session_recall import EARLIER_TURNS_LOOKED_AT
from pilot.system import sysinfo


class FailingModelRouter:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self.generate_calls += 1
        raise AssertionError("CPU usage fast path should not call the model")


class TrackingMemory:
    def __init__(self) -> None:
        self.context_calls = 0

    async def get_context(self, query: str) -> str:  # noqa: ARG002
        self.context_calls += 1
        return ""


class PlanningMemory:
    def __init__(self) -> None:
        self.context_kwargs: dict[str, Any] = {}
        self.history_kwargs: dict[str, Any] = {}

    async def get_context(self, query: str, **kwargs: Any) -> str:  # noqa: ARG002
        self.context_kwargs = kwargs
        return ""

    async def get_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.history_kwargs = kwargs
        return []


class PlanningModel:
    def __init__(self) -> None:
        self._config = SimpleNamespace(
            memory=SimpleNamespace(
                max_context_tokens=3000,
                max_recent_messages=7,
            )
        )

    async def generate(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        return '{"explanation":"No action needed.","actions":[]}'


@pytest.mark.asyncio
async def test_cpu_usage_query_uses_local_fast_path() -> None:
    model = FailingModelRouter()
    memory = TrackingMemory()
    planner = Planner(model, memory)  # type: ignore[arg-type]

    plan = await planner.plan("What's my CPU usage?")

    assert plan.error is None
    assert [action.action_type for action in plan.actions] == [ActionType.CPU_USAGE]
    assert model.generate_calls == 0
    assert memory.context_calls == 0


@pytest.mark.asyncio
async def test_cpu_usage_uses_short_sample_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    intervals: list[float | None] = []

    async def fake_cpu_info(sample_interval: float | None = 0.5) -> str:
        intervals.append(sample_interval)
        return "=== CPU ===\n  Average usage: 12.3%"

    monkeypatch.setattr(sysinfo, "_cpu_info", fake_cpu_info)

    output = await sysinfo.cpu_usage()

    assert "Average usage" in output
    assert intervals == [sysinfo.CPU_USAGE_SAMPLE_SECONDS]
    assert sysinfo.CPU_USAGE_SAMPLE_SECONDS <= 0.02


@pytest.mark.asyncio
async def test_system_probe_preparation_primes_cpu_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    primed = False

    def prime() -> None:
        nonlocal primed
        primed = True

    monkeypatch.setattr(sysinfo, "_prime_psutil_cpu", prime)

    await sysinfo.prepare_system_probes()

    assert primed is True


@pytest.mark.asyncio
async def test_comprehensive_system_probes_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    started: set[str] = set()
    all_started = asyncio.Event()

    def probe(name: str):
        async def run() -> str:
            started.add(name)
            if len(started) == 4:
                all_started.set()
            await all_started.wait()
            return f"=== {name} ==="

        return run

    monkeypatch.setattr(sysinfo, "_cpu_info", probe("cpu"))
    monkeypatch.setattr(sysinfo, "_memory_info", probe("memory"))
    monkeypatch.setattr(sysinfo, "_disk_info", probe("disk"))
    monkeypatch.setattr(sysinfo, "_network_info", probe("network"))

    output = await asyncio.wait_for(
        sysinfo.system_info(["cpu", "memory", "disk", "network"]),
        timeout=0.2,
    )

    assert started == {"cpu", "memory", "disk", "network"}
    assert output.index("cpu") < output.index("memory") < output.index("disk") < output.index("network")


@pytest.mark.asyncio
async def test_full_latency_benchmark_completes_without_model_or_thread_leak() -> None:
    timings, model_calls = await asyncio.wait_for(benchmark(1), timeout=10)

    assert len(timings) == 1
    assert timings[0] > 0
    assert model_calls == 0


@pytest.mark.asyncio
async def test_latency_report_separates_ready_cold_and_steady_state() -> None:
    report = await asyncio.wait_for(benchmark_report(3, warmup=1), timeout=10)

    assert report["cold_ready_request_ms"] > 0
    assert report["steady_state"]["iterations"] == 3
    assert report["steady_state"]["p95_ms"] >= report["steady_state"]["median_ms"]
    assert report["model_generate_calls"] == 0
    assert report["total_model_generate_calls"] == 0


def test_latency_summary_reports_tail_and_variance() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0, 20.0])

    assert summary["iterations"] == 5
    assert summary["median_ms"] == 3.0
    assert summary["p95_ms"] > summary["median_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]
    assert summary["stdev_ms"] > 0


@pytest.mark.asyncio
async def test_local_status_suite_executes_real_fast_paths_without_model() -> None:
    report = await asyncio.wait_for(benchmark_suite(2, warmup=1), timeout=10)

    assert set(report["scenarios"]) == {
        "cpu_usage",
        "memory_usage",
        "disk_usage",
        "system_information",
    }
    assert report["model_generate_calls"] == 0
    assert all(scenario["iterations"] == 2 for scenario in report["scenarios"].values())


@pytest.mark.asyncio
async def test_planner_bounds_memory_and_raw_history() -> None:
    memory = PlanningMemory()
    planner = Planner(PlanningModel(), memory)  # type: ignore[arg-type]

    plan = await planner.plan(
        "Analyze the project dependency graph and explain the result",
        session_id="chat-42",
    )

    assert plan.error is None
    assert memory.context_kwargs == {
        "session_id": "chat-42",
        "max_tokens": 1000,
    }
    # Six recent turns go to the model as messages; older ones are read only to pick what still matters for
    # "Earlier in this conversation" (pilot.memory.session_recall).
    assert memory.history_kwargs == {
        "limit": 6 + EARLIER_TURNS_LOOKED_AT,
        "session_id": "chat-42",
    }


@pytest.mark.asyncio
async def test_a_long_session_still_sends_six_raw_turns_and_recalls_the_rest_as_context() -> None:
    class LongMemory(PlanningMemory):
        async def get_history(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.history_kwargs = kwargs
            turns = [{"user_input": "Today's report is for the Porto warehouse", "explanation": "Got it."}]
            turns += [{"user_input": f"Step {i}", "explanation": f"Done {i}."} for i in range(19)]
            return list(reversed(turns))  # newest first, like the store

    class KeepingModel(PlanningModel):
        sent: list[dict[str, Any]] = []

        async def generate(self, messages, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
            KeepingModel.sent = messages
            return '{"explanation":"No action needed.","actions":[]}'

    planner = Planner(KeepingModel(), LongMemory())  # type: ignore[arg-type]
    await planner.plan("Email the warehouse report to the Porto team", session_id="chat-7")
    history = [m for m in KeepingModel.sent[1:-1] if m["role"] == "user"]
    assert 0 < len(history) <= 6 and not any("Porto" in m["content"] for m in history)
    assert "Porto warehouse" in KeepingModel.sent[-1]["content"]  # recalled into the context block
