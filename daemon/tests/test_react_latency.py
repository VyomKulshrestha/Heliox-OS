"""Regression tests for simple ReAct loop latency optimizations."""

from __future__ import annotations

from typing import Any

import pytest

from pilot.actions import ActionType
from pilot.agents.planner import Planner
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
    assert intervals == [0.1]
