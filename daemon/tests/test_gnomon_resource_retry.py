"""Gnomon waits for memory on its own after a load fails for lack of it, and says what is missing."""

import asyncio

import pytest

from pilot.intelligence import gnomon_reviewer
from pilot.intelligence.gnomon_reviewer import GIB, GnomonReviewer, classify_resource_shortage

RAM_ERROR = "At least 2.5 GiB of available system RAM is required before local model loading"
VRAM_ERROR = "At least 4 GiB of free CUDA memory is required for local model scoring"
TICK = 0.02


class _Engine:
    """Fails to load while `fail` is set; counts every load attempt."""

    def __init__(self, error: str | BaseException | None = RAM_ERROR) -> None:
        self.error = error
        self.fail = error is not None
        self.loads = 0
        self.loaded = False

    def load(self):
        self.loads += 1
        if self.fail:
            raise self.error if isinstance(self.error, BaseException) else RuntimeError(self.error)
        self.loaded = True

    def unload(self):
        self.loaded = False

    def score(self, state, kind, question, options):
        return {"status": "ok", "answer": {"yes_probability": 0.9}}


@pytest.fixture
def free_memory(monkeypatch):
    """Free RAM / VRAM as the reviewer sees them, in GiB."""
    memory = {"ram": 1.8, "gpu": None}
    monkeypatch.setattr(gnomon_reviewer, "_free_host_memory", lambda: int(memory["ram"] * GIB))
    monkeypatch.setattr(
        gnomon_reviewer, "_free_gpu_memory", lambda: None if memory["gpu"] is None else int(memory["gpu"] * GIB)
    )
    return memory


async def _until(predicate, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


async def _failed(engine) -> GnomonReviewer:
    reviewer = GnomonReviewer(lambda: engine, resource_check_seconds=TICK)
    reviewer.warm()
    assert await _until(lambda: reviewer.status()["state"] == "unavailable")
    return reviewer


async def test_recovers_once_memory_frees_without_a_new_request(free_memory):
    engine = _Engine()
    reviewer = await _failed(engine)

    await asyncio.sleep(TICK * 10)  # memory still short: checks only, no load attempts
    assert engine.loads == 1
    assert reviewer.status()["state"] == "unavailable"

    engine.fail = False
    free_memory["ram"] = 6.0
    assert await _until(lambda: reviewer.status()["state"] == "ready")
    assert engine.loads == 2
    status = reviewer.status()
    assert status["reason"] == "" and status["last_error"] == "" and status["waiting_for"] is None
    await reviewer.close()


async def test_no_retry_storm_when_loading_keeps_failing(free_memory):
    free_memory["ram"] = 6.0  # the cheap check passes, yet the load still fails (another process grabbed it)
    engine = _Engine()
    reviewer = await _failed(engine)

    await asyncio.sleep(TICK * 25)

    # Each failed attempt doubles the wait (1, 2, 4, 8, 16 ticks): a handful of loads, not one per tick.
    assert 2 <= engine.loads <= 6
    await reviewer.close()


async def test_close_cancels_the_memory_watch(free_memory):
    engine = _Engine()
    reviewer = await _failed(engine)
    assert await _until(lambda: reviewer.status()["retrying"])
    task = reviewer._resource_task

    await reviewer.close()
    engine.fail = False
    free_memory["ram"] = 6.0
    await asyncio.sleep(TICK * 5)

    assert task.done()
    assert engine.loads == 1
    assert reviewer.status()["state"] == "unavailable"
    assert reviewer.status()["retrying"] is False


async def test_unload_cancels_the_memory_watch(free_memory):
    engine = _Engine()
    reviewer = await _failed(engine)
    assert await _until(lambda: reviewer.status()["retrying"])

    await reviewer.unload()
    engine.fail = False
    free_memory["ram"] = 6.0
    await asyncio.sleep(TICK * 5)

    assert engine.loads == 1
    await reviewer.close()


async def test_other_failures_keep_the_cool_down_and_do_not_retry_on_their_own(free_memory):
    engine = _Engine(FileNotFoundError("Missing local model artifact: head.safetensors"))
    reviewer = await _failed(engine)
    free_memory["ram"] = 6.0

    await asyncio.sleep(TICK * 10)
    reviewer.warm()  # inside the cool-down

    assert engine.loads == 1
    status = reviewer.status()
    assert status["state"] == "unavailable"
    assert status["retrying"] is False and status["waiting_for"] is None
    assert status["reason"] == "Could not load: Missing local model artifact: head.safetensors"
    await reviewer.close()


async def test_status_says_what_memory_is_missing(free_memory):
    engine = _Engine()
    reviewer = await _failed(engine)
    assert await _until(lambda: reviewer.status()["retrying"])

    status = reviewer.status()
    assert status["state"] == "unavailable"
    assert status["execution_authority"] is False
    assert "2.5 GiB" in status["last_error"]
    assert status["reason"] == "Needs 2.5 GB of free memory, 1.8 GB free now. Tries again on its own."
    assert status["waiting_for"] == {"resource": "memory", "required_gb": 2.5, "free_gb": 1.8}

    free_memory["ram"] = 2.1  # still short; the status follows the live number
    assert reviewer.status()["reason"].startswith("Needs 2.5 GB of free memory, 2.1 GB free now")
    await reviewer.close()


async def test_status_for_graphics_memory(free_memory):
    free_memory["gpu"] = 2.1
    engine = _Engine(VRAM_ERROR)
    reviewer = await _failed(engine)

    assert reviewer.status()["reason"].startswith("Needs 4 GB of free graphics memory, 2.1 GB free now.")

    free_memory["gpu"] = 5.0
    engine.fail = False
    free_memory["ram"] = 6.0
    assert await _until(lambda: reviewer.status()["state"] == "ready")
    await reviewer.close()


def test_classify_resource_shortage():
    ram = classify_resource_shortage(RuntimeError(RAM_ERROR))
    assert ram is not None and ram.resource == "memory" and ram.required_bytes == int(2.5 * GIB)
    vram = classify_resource_shortage(RuntimeError(VRAM_ERROR))
    assert vram is not None and vram.resource == "graphics memory" and vram.required_bytes == 4 * GIB
    oom = classify_resource_shortage(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"))
    assert oom is not None and oom.resource == "graphics memory" and oom.required_bytes is None
    assert classify_resource_shortage(RuntimeError("CUDA is required for this opt-in local research model")) is None
    assert classify_resource_shortage(FileNotFoundError("Missing local model artifact")) is None
