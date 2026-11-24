"""Work no one waits for yields to heat: on a card near its limit the voice's fixed sentences made ahead and the next
sentence before its turn wait until it cools, while a reply a person waits for still goes ahead. Each card is judged by
its own slowdown and maximum temperatures where the driver gives them, with a fixed fallback.

On one development laptop an idle card at 77 °C reached 82 °C within 40 s while the voice made 202 fixed sentences
ahead on it. Made-up machines and a made-up clock only: no card is read.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.system import local_tts_worker
from pilot.system.gpu_arbiter import (
    BACKGROUND_MARGIN_C,
    BACKGROUND_RESUME_C,
    CARD_MARGIN_C,
    DEFAULT_MAX_TEMPERATURE_C,
    GIB,
    CardTemperatures,
    GpuArbiter,
    GpuInfo,
    MachineReading,
    background_limit_c,
    parse_card_temperatures,
)
from pilot.system.local_tts_worker import LocalTTSWorkerManager, SpeechStore

SMI_TEMPERATURE = """
==============NVSMI LOG==============

Attached GPUs                             : 2
GPU 00000000:01:00.0
    Temperature
        GPU Current Temp                  : 61 C
        GPU T.Limit Temp                  : 26 C
        GPU Shutdown Temp                 : 102 C
        GPU Slowdown Temp                 : 87 C
        GPU Max Operating Temp            : 87 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A

GPU 00000000:02:00.0
    Temperature
        GPU Current Temp                  : 40 C
        GPU Shutdown Temp                 : N/A
        GPU Slowdown Temp                 : N/A
        GPU Max Operating Temp            : N/A
"""


def test_each_card_s_own_thresholds_are_read_and_a_missing_one_is_none():
    first, second = parse_card_temperatures(SMI_TEMPERATURE)
    assert first == CardTemperatures(slowdown_c=87.0, max_operating_c=87.0, shutdown_c=102.0)
    assert second == CardTemperatures()


def test_the_limit_is_the_card_s_own_less_a_margin_with_a_fixed_fallback():
    default = DEFAULT_MAX_TEMPERATURE_C - BACKGROUND_MARGIN_C
    assert background_limit_c(DEFAULT_MAX_TEMPERATURE_C, None) == default  # nothing reported: the setting
    assert background_limit_c(DEFAULT_MAX_TEMPERATURE_C, CardTemperatures()) == default
    laptop = CardTemperatures(slowdown_c=87.0, max_operating_c=87.0)  # slows itself early: judged by its own figure
    assert background_limit_c(DEFAULT_MAX_TEMPERATURE_C, laptop) == 87.0 - CARD_MARGIN_C - BACKGROUND_MARGIN_C
    desktop = CardTemperatures(slowdown_c=95.0, max_operating_c=93.0)  # runs hotter: the setting still caps it
    assert background_limit_c(DEFAULT_MAX_TEMPERATURE_C, desktop) == default
    assert background_limit_c(DEFAULT_MAX_TEMPERATURE_C, CardTemperatures(slowdown_c=0.0)) == default  # not real


def _machine(temperature: dict[str, float], own: tuple[CardTemperatures, ...] = ()) -> GpuArbiter:
    def read() -> MachineReading:
        card = GpuInfo(0, "RTX", temperature["now"], 6 * GIB, 8 * GIB)
        return MachineReading("nvidia", (card,), 8 * GIB, 16 * GIB, "")

    return GpuArbiter(environ={}, read=read, holder=lambda: None, temperatures=lambda: own, cache_seconds=0.0)


def test_work_ahead_pauses_near_the_limit_and_goes_on_once_the_card_has_cooled():
    temperature = {"now": 70.0}
    arbiter = _machine(temperature)
    limit = DEFAULT_MAX_TEMPERATURE_C - BACKGROUND_MARGIN_C
    assert arbiter.background_heat() == ""
    temperature["now"] = limit  # 77 °C on the default
    assert "77" in arbiter.background_heat()
    temperature["now"] = limit - BACKGROUND_RESUME_C + 0.5  # a little cooler: still paused, no flapping
    assert arbiter.background_heat()
    temperature["now"] = limit - BACKGROUND_RESUME_C - 0.5
    assert arbiter.background_heat() == ""
    temperature["now"] = 81.0  # on a card that slows itself at 87 °C, 81 °C is too warm for work ahead
    assert _machine(temperature, (CardTemperatures(slowdown_c=87.0),)).background_heat()


class Worker:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.returncode: int | None = None
        self._payload = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, close=lambda: None)

    def _write(self, payload: bytes) -> None:
        self._payload = payload

    def _close(self) -> None:
        self.stdin.closed = True
        self.returncode = 0

    def _readline(self) -> bytes:
        request = json.loads(self._payload)
        if request.get("op") == "load":
            return b'{"status":"ok","op":"load"}\n'
        self.asked.append(request["text"])
        Path(request["output_file"]).write_bytes(b"RIFF-audio")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_fixed_sentences_wait_while_the_card_is_warm_and_a_reply_does_not(monkeypatch, tmp_path):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    worker, clock = Worker(), Clock()
    manager = LocalTTSWorkerManager(
        spawn=lambda engine, device, gpu_lease=False: worker, clock=clock, sleep=clock.sleep
    )
    manager._start_idle_watch = lambda: None  # type: ignore[method-assign]  # the made-up clock is not the idle timer's
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    await manager.ensure_ready("kokoro_tts", "af_heart")
    manager.gpu_lease = True  # the worker holds the card
    warm = {"reason": "The graphics card is at 79 °C."}
    manager.heat_reason = lambda: warm["reason"]
    clock.now += local_tts_worker.MAKE_AHEAD_IDLE_SECONDS + 1
    making = asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", ["One.", "Two."], battery=lambda: False, poll_seconds=1.0)
    )
    for _ in range(50):
        await asyncio.sleep(0)
    assert worker.asked == []  # nothing made ahead on a warm card

    async def play(piece) -> None:
        return None

    await manager.speak("kokoro_tts", "A reply.", "af_heart", None, play)
    assert worker.asked == ["A reply."]  # a person is waiting: it goes ahead
    clock.now += local_tts_worker.MAKE_AHEAD_IDLE_SECONDS + 1
    warm["reason"] = ""  # the card has cooled
    assert await making == 2
    assert worker.asked == ["A reply.", "One.", "Two."]
    manager.gpu_lease = False  # on the processor the card's heat is not asked at all
    manager.heat_reason = lambda: (_ for _ in ()).throw(AssertionError("asked"))
    assert await manager.too_warm_now() == ""
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_next_sentence_waits_for_a_warm_card_unless_a_reply_wants_it(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    worker, clock = Worker(), Clock()
    manager = LocalTTSWorkerManager(
        spawn=lambda engine, device, gpu_lease=False: worker, clock=clock, sleep=clock.sleep
    )
    manager._start_idle_watch = lambda: None  # type: ignore[method-assign]  # the made-up clock is not the idle timer's
    await manager.ensure_ready("kokoro_tts", "af_heart")
    manager.gpu_lease = True
    manager.heat_reason = lambda: "The graphics card is at 79 °C."

    async def play(piece) -> None:
        return None

    manager.prepare("kokoro_tts", "The next one.", "af_heart")
    await manager.speak("kokoro_tts", "This one.", "af_heart", None, play)
    for _ in range(50):
        await asyncio.sleep(0)
    assert worker.asked == ["This one."]  # its turn came, but the card is warm and nobody waits for it yet
    await manager.speak("kokoro_tts", "The next one.", "af_heart", None, play)  # now a reply waits for it
    assert worker.asked == ["This one.", "The next one."]
    await manager.shutdown()
