"""From process start to every late part started, how long any one step holds the event loop: a real start-up
(initialize, ready, the learner and the late parts) in asyncio's debug mode. After ready no step may hold the loop
longer than AFTER_READY_BOUND: those first seconds are when a person speaks first, and imports of the learner's and
neural input's libraries used to hold it for over a second. Steps before ready are printed, not asserted: while
starting, ping and startup_status wait for them (docs/RELIABILITY.md).

The bound is an order of magnitude for this kind of machine, not a quoted time: debug mode itself adds overhead, and
a shared or busy machine adds more."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer

_spec = importlib.util.spec_from_file_location("loop_watch", Path(__file__).with_name("loop_watch.py"))
loop_watch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["loop_watch"] = loop_watch
_spec.loader.exec_module(loop_watch)
watch, unwatch = loop_watch.watch, loop_watch.unwatch

AFTER_READY_BOUND = 0.1  # seconds: no step the watcher flags (it flags 100 ms and over); see the module docstring
LIMIT = 600.0  # failure limit only


@pytest.mark.asyncio
async def test_after_ready_no_step_holds_the_loop_for_long(unused_tcp_port, tmp_path, monkeypatch):
    for name in ("config", "data", "state", "runtime"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr("pilot.config.CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr("pilot.config.CONFIG_FILE", tmp_path / "config" / "config.toml")
    monkeypatch.setattr("pilot.config.RESTRICTIONS_FILE", tmp_path / "config" / "restrictions.toml")
    monkeypatch.setattr("pilot.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("pilot.server.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("pilot.config.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("pilot.config.DB_FILE", tmp_path / "data" / "pilot.db")
    monkeypatch.setattr("pilot.config.LOG_FILE", tmp_path / "state" / "pilot.log")
    monkeypatch.setattr("pilot.security.vault.KeyVault.available", property(lambda _self: False))  # no real keyring
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", unused_tcp_port, "test-token"

    ready_at: list[float] = []
    original = PilotServer._begin_after_ready

    def begin_after_ready(self):
        ready_at.append(time.time())
        original(self)

    monkeypatch.setattr(PilotServer, "_begin_after_ready", begin_after_ready)
    collector = watch(asyncio.get_running_loop(), threshold=0.1)
    try:
        with (
            patch("pilot.models.router.ModelRouter.initialize", new_callable=AsyncMock),
            patch("pilot.models.cache.LLMCache.initialize", new_callable=AsyncMock),
            patch("pilot.memory.store.MemoryStore.initialize", new_callable=AsyncMock),
            patch("pilot.agents.screen_vision.ScreenVisionAgent.start", new_callable=AsyncMock),
            patch("pilot.cognitive.cognitive_engine.CognitiveEngine.load_model", new_callable=AsyncMock),
            patch("pilot.security.audit.AuditLogger", return_value=MagicMock()),
        ):
            server = PilotServer(config)
            await asyncio.wait_for(server.start(), LIMIT)
            late = server._late_parts()
            deadline = time.monotonic() + LIMIT
            while time.monotonic() < deadline and any(
                state not in ("ready", "failed") for state in late.snapshot().values()
            ):
                await asyncio.sleep(0.1)
            assert all(state in ("ready", "failed") for state in late.snapshot().values()), late.snapshot()
            await server.stop()
    finally:
        unwatch(collector)

    assert ready_at, "never reached ready"
    before = [step for step in collector.steps if step.at < ready_at[0]]
    after = [step for step in collector.steps if step.at >= ready_at[0]]
    for step in sorted(before, key=lambda step: step.seconds, reverse=True):
        print(f"before ready: {step.seconds:.3f} s  {step.step[:200]}")
    for step in sorted(after, key=lambda step: step.seconds, reverse=True):
        print(f"after ready: {step.seconds:.3f} s  {step.step[:200]}")
    longest = max((step.seconds for step in after), default=0.0)
    assert longest <= AFTER_READY_BOUND, [f"{step.seconds:.3f} s {step.step[:300]}" for step in after]
