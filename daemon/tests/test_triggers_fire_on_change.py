"""A trigger whose condition holds over time fires when it starts to hold, not on every look while it holds and not
again after a restart while it still holds. The condition is a fake the test sets; no system metric is read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.system import triggers
from pilot.system.triggers import TriggerEngine


class _Condition:
    def __init__(self, holds: bool) -> None:
        self.holds = holds
        self.looks = 0

    async def __call__(self, _condition: dict) -> bool:  # set on the class, not bound: no engine argument
        self.looks += 1
        return self.holds


@pytest.fixture
def cpu_high(monkeypatch: pytest.MonkeyPatch) -> _Condition:
    condition = _Condition(True)
    monkeypatch.setattr(TriggerEngine, "_check_cpu", condition)
    return condition


def _engine(store: Path) -> TriggerEngine:
    engine = TriggerEngine()
    engine.attach_store(store)
    return engine


@pytest.mark.asyncio
async def test_a_condition_that_still_holds_does_not_fire_again_after_two_restarts(
    tmp_path: Path, cpu_high: _Condition
) -> None:
    store = tmp_path / "triggers.json"
    first = _engine(store)
    made = first.create_trigger("busy", "cpu_threshold", {"threshold": 90}, "tell me the CPU is busy", 0, 0)

    assert [trigger.id for trigger in await first.evaluate_once()] == [made.id]  # it started to hold: fires once
    assert await first.evaluate_once() == []  # still holds: nothing new

    second = _engine(store)  # the daemon restarted
    assert await second.evaluate_once() == []
    third = _engine(store)  # and again
    assert await third.evaluate_once() == []
    assert cpu_high.looks == 4
    assert third.get_trigger(made.id).fire_count == 1

    # A real change fires again, and that is kept across a restart too.
    cpu_high.holds = False
    assert await third.evaluate_once() == []
    cpu_high.holds = True
    assert [trigger.id for trigger in await third.evaluate_once()] == [made.id]
    fourth = _engine(store)
    assert await fourth.evaluate_once() == []
    assert fourth.get_trigger(made.id).fire_count == 2


@pytest.mark.asyncio
async def test_a_condition_that_stopped_holding_while_stopped_fires_when_it_holds_again(
    tmp_path: Path, cpu_high: _Condition
) -> None:
    store = tmp_path / "triggers.json"
    engine = _engine(store)
    engine.create_trigger("busy", "cpu_threshold", {"threshold": 90}, "tell me", 0, 0)
    await engine.evaluate_once()

    cpu_high.holds = False
    restarted = _engine(store)
    assert await restarted.evaluate_once() == []  # the store now remembers it stopped holding
    stored = json.loads(store.read_text(encoding="utf-8"))["triggers"][0]
    assert stored["condition_held"] is False

    cpu_high.holds = True
    assert len(await _engine(store).evaluate_once()) == 1


@pytest.mark.asyncio
async def test_a_store_written_before_this_field_fires_once_then_remembers(
    tmp_path: Path, cpu_high: _Condition
) -> None:
    store = tmp_path / "triggers.json"
    engine = _engine(store)
    engine.create_trigger("busy", "cpu_threshold", {"threshold": 90}, "tell me", 0, 0)
    data = json.loads(store.read_text(encoding="utf-8"))
    del data["triggers"][0]["condition_held"]  # as an older Heliox wrote it
    store.write_text(json.dumps(data), encoding="utf-8")

    assert len(await _engine(store).evaluate_once()) == 1
    assert await _engine(store).evaluate_once() == []


@pytest.mark.asyncio
async def test_a_time_interval_still_fires_by_its_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path / "triggers.json")
    engine.create_trigger("tick", "time_interval", {}, "say tick", 0, 0)

    assert len(await engine.evaluate_once()) == 1
    assert len(await engine.evaluate_once()) == 1  # an interval is not a state: each due look fires
    assert triggers.TriggerType.TIME_INTERVAL not in triggers.LEVEL_TRIGGERS
