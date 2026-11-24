"""One voice per reply, with a reply bounded by what it is: each thing Heliox says through the speech coordinator is its
own reply (one answer of the conversation layer with its holding phrase, one Read-aloud call, one reminder), and no
caller can keep Heliox in the system voice once its own voice is ready.

On 24 November 60 Read-aloud calls about 6 s apart were all spoken in the system voice for 5.6 minutes after Heliox's
own voice had loaded: the only end of a reply was 6 s of silence, and they never left one. Fake worker, made-up
clock, counted calls; no model, no audio device.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.system import local_tts_worker
from pilot.system.companion_speech import CompanionSpeechCoordinator, reply_name
from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady


class SlowWorker:
    """Loads only once ``loaded`` is set; speaks at once (the audio as a file)."""

    def __init__(self) -> None:
        self.loaded = threading.Event()
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
            self.loaded.wait(10)
            return b'{"status":"ok","op":"load"}\n'
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


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker, clock = SlowWorker(), Clock()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: worker, clock=clock)
    manager.environ = {}
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    voices: list[str] = []

    async def speaker(text: str, recorder) -> bool:
        """The voice's own rule, as voice.speak applies it: Heliox's voice, or the system voice for this text."""
        try:
            await manager.speak("kokoro_tts", text, "af_heart", None, play)
            voices.append("local")
        except LocalVoiceNotReady:
            voices.append("system")
            manager.note_spoken()  # the system voice said it
        return False

    async def play(piece) -> None:
        return None

    coordinator = CompanionSpeechCoordinator(speaker=speaker, dedupe_window_seconds=0.0)
    return SimpleNamespace(worker=worker, clock=clock, manager=manager, voices=voices, coordinator=coordinator)


async def ready(item) -> None:
    item.worker.loaded.set()
    for _ in range(500):
        if item.manager.loaded:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the voice never finished loading")


@pytest.mark.asyncio
async def test_sixty_read_aloud_calls_use_heliox_s_voice_from_the_one_after_it_is_ready(setup):
    item = setup
    for number in range(60):
        if number == 3:
            await ready(item)
        await item.coordinator.speak(f"Line {number} of the page.", dedupe_key=f"read:{number}")
        item.clock.now += 5.0  # never 6 s of silence between them
    assert item.voices[:3] == ["system"] * 3
    assert item.voices[3:] == ["local"] * 57
    await item.manager.shutdown()


@pytest.mark.asyncio
async def test_a_caller_that_names_no_reply_cannot_keep_the_system_voice(setup):
    """speak() with no reply name (a direct caller) falls back on silence to end a reply, and once Heliox's voice is
    ready its reply says at most SYSTEM_PIECES_AFTER_READY more pieces in the system voice."""
    item = setup

    async def say(text: str) -> str:
        try:
            await item.manager.speak("kokoro_tts", text, "af_heart", None, lambda piece: asyncio.sleep(0))
            return "local"
        except LocalVoiceNotReady:
            item.manager.note_spoken()
            return "system"

    said = []
    for number in range(20):
        if number == 3:
            await ready(item)
        said.append(await say(f"Progress line {number}."))
        item.clock.now += 5.0
    bound = local_tts_worker.SYSTEM_PIECES_AFTER_READY
    assert said == ["system"] * (3 + bound) + ["local"] * (20 - 3 - bound)
    await item.manager.shutdown()


@pytest.mark.asyncio
async def test_an_answer_that_began_in_the_system_voice_finishes_in_it_and_the_next_answer_is_heliox_s(setup):
    item = setup
    await item.coordinator.speak("Let me check.", channel="task_narration", dedupe_key="conv-hold:7")
    await ready(item)  # the voice is ready while the answer is being said
    for number in range(3):
        item.clock.now += 1.0
        await item.coordinator.speak(f"Sentence {number} of the answer.", dedupe_key=f"conv:7:{number}")
    assert item.voices == ["system"] * 4  # the holding phrase and every sentence of its answer: one voice
    await item.coordinator.speak("The next answer.", dedupe_key="conv:8:0")
    assert item.voices[-1] == "local"
    await item.manager.shutdown()


@pytest.mark.asyncio
async def test_a_hum_is_said_only_when_the_answer_after_it_will_be_in_the_same_voice(setup):
    item = setup
    item.manager.begin_reply()
    assert not item.manager.speaks_locally_now("kokoro_tts")  # not loaded: no hum (it would be Heliox's voice)
    await item.coordinator.speak("Let me check.", channel="task_narration", dedupe_key="conv-hold:1")
    await ready(item)
    # The next turn: the system-voice answer before it does not stop the hum, and the hum and its answer are both
    # Heliox's voice.
    assert item.manager.speaks_locally_now("kokoro_tts")
    await item.coordinator.speak("Mm.", channel="task_narration", dedupe_key="conv-ack:turn-2")
    await item.coordinator.speak("Here is the answer.", dedupe_key="conv:2:0")
    assert item.voices == ["system", "local", "local"]
    await item.manager.shutdown()


def test_the_conversation_layer_s_keys_name_one_reply_per_answer():
    assert reply_name("conv:140:0", 1) == reply_name("conv:140:3", 2) == reply_name("conv-hold:140", 3) == "conv:140"
    assert reply_name("conv-ack:140", 4) == reply_name("conv-sorry:140", 5) == "conv:140"
    assert reply_name("conv:141:0", 6) != "conv:140"
    assert reply_name("read:1", 7) == "speech:7" and reply_name("", 8) == "speech:8"
    assert reply_name("conv-progress:12.5:2", 9) == "speech:9"  # each progress line of a task is its own
