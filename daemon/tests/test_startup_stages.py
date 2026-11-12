"""The start-up stage counter reads PilotServer.initialize as it is: the stages in order, and what each awaits."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("startup_stages", ROOT / "benchmarks" / "startup_stages.py")
startup_stages = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = startup_stages  # its dataclasses look their module up
spec.loader.exec_module(startup_stages)


def test_the_stages_are_read_in_the_order_the_window_sees_them():
    found = startup_stages.stages()
    assert [stage.name for stage in found] == [
        startup_stages.BEFORE_FIRST,
        "connecting the language model",
        "opening Heliox's records",
        "loading memory",
        "preparing the planner",
        "starting Live, reminders and the timeline",
    ]


def test_each_stage_lists_the_work_the_first_request_waits_for():
    by_name = {stage.name: stage for stage in startup_stages.stages()}
    assert "model_router.initialize" in by_name["connecting the language model"].awaited
    assert "self._durable_tasks.initialize" in by_name["opening Heliox's records"].awaited
    memory = by_name["loading memory"]
    assert "self._online_learning.initialize" in memory.awaited  # replays the experience ledger
    assert "pilot.intelligence.online_learning" in memory.imports
    gated = [call for call in by_name["starting Live, reminders and the timeline"].awaited if "[only if" in call]
    assert any("cognitive.enabled" in call for call in gated)
