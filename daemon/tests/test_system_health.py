"""Health check answers: computed from one reading, never closing anything."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.system_health import (
    SystemHealth,
    describe_cpu,
    describe_disks,
    describe_health,
    describe_memory,
    read_snapshot,
)

GB = 1024**3


def snapshot(cpu=12.0, memory_percent=50.0, free_gb=200.0, battery=None):
    return {
        "at": 0.0,
        "cpu": cpu,
        "memory": {"total": 16 * GB, "used": 16 * GB * memory_percent / 100, "percent": memory_percent},
        "disks": [
            {"drive": "C:\\", "total": 500 * GB, "free": free_gb * GB},
            {"drive": "/", "total": 100 * GB, "free": 60 * GB},
        ],
        "battery": battery,
        "apps": [
            {"app": "Chrome", "cpu": 40.0, "memory": 3 * GB, "processes": 13},
            {"app": "Code", "cpu": 2.0, "memory": 1.2 * GB, "processes": 2},
            {"app": "Teams", "cpu": 0.5, "memory": 0.8 * GB, "processes": 1},
        ],
    }


def test_healthy_computer():
    assert describe_health(snapshot()) == (
        "Nothing looks overloaded: the processor is at 12% and memory at 50%. "
        "The biggest apps are Chrome 3.0 GB across 13 processes, Code 1.2 GB, Teams 0.8 GB."
    )


def test_full_memory_and_busy_processor_name_the_culprit():
    text = describe_health(snapshot(cpu=92.0, memory_percent=91.0))
    assert text.startswith("Memory is nearly full (91%, 15 GB of 16 GB), which is the most likely reason")
    assert "the processor is busy at 92%, mostly Chrome (40%)" in text
    assert text.endswith('say "close Chrome" and I\'ll ask before closing it.')


def test_low_disk_and_battery_are_mentioned():
    text = describe_health(snapshot(free_gb=10.0, battery={"percent": 12.0, "plugged": False}))
    assert text.startswith("C: is almost full (10 GB free); the battery is at 12% and may be saving power.")


def test_memory_cpu_and_disk_answers():
    assert describe_memory(snapshot()) == (
        "Memory is 50% used (8.0 GB of 16 GB). Most of it: Chrome 3.0 GB across 13 processes, Code 1.2 GB, Teams 0.8 GB."
    )
    assert describe_cpu(snapshot(cpu=55.0)) == "The processor is at 55%. Busiest: Chrome 40%, Code 2%."
    quiet = snapshot(cpu=3.0)
    for app in quiet["apps"]:
        app["cpu"] = 0.2
    assert describe_cpu(quiet) == "The processor is at 3%, and no app is using much of it."
    assert describe_disks(snapshot(free_gb=20.0)) == (
        "C: has 20 GB free of 500 GB (4%); / has 60 GB free of 100 GB (60%). "
        "That is getting full; clearing Downloads or old installers would help."
    )


@pytest.mark.asyncio
async def test_questions_are_answered_and_others_left_alone():
    health = SystemHealth(read=snapshot)
    assert (await health.answer_async("Why is my computer so slow?")).startswith("Nothing looks overloaded")
    assert (await health.answer_async("what's slowing down my laptop")).startswith("Nothing looks overloaded")
    assert (await health.answer_async("What's using all my RAM?")).startswith("Memory is 50% used")
    assert (await health.answer_async("what's using my CPU")).startswith("The processor is at 12%")
    assert (await health.answer_async("how much disk space do I have left")).startswith("C: has 200 GB free")
    assert (await health.answer_async("is my drive full")).startswith("C: has")
    for text in ("why is the internet slow", "what's using port 8080", "open task manager", "close chrome"):
        assert await health.answer_async(text) is None, text


@pytest.mark.asyncio
async def test_read_failure_is_reported():
    def broken():
        raise OSError("access denied")

    assert await SystemHealth(read=broken).answer_async("why is my pc slow") == (
        "I couldn't read how this computer is doing: access denied"
    )


def test_real_reading_has_every_field():
    reading = read_snapshot(sample_seconds=0.05)
    assert set(reading) == {"at", "cpu", "memory", "disks", "battery", "apps"}
    assert reading["memory"]["total"] > 0
    assert all({"app", "cpu", "memory", "processes"} <= set(app) for app in reading["apps"])


@pytest.mark.asyncio
async def test_system_health_rpc():
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._system_health = SystemHealth(read=snapshot)

    result = await server._handle_system_health({}, MagicMock())
    assert result["status"] == "ok" and result["cpu"] == 12.0
    assert result["summary"].startswith("Nothing looks overloaded")
    assert (await server._control_reply("how is my computer doing")).startswith("Nothing looks overloaded")
