"""The fixed sentences Heliox says word for word are gathered from where they are written, made ahead only while the
voice is idle and nobody talks to Heliox, and played from the store on an exact match of text, voice and settings.
Fake workers only: no model, no audio device."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from pilot.conversation import handover, layer, say_back
from pilot.system import fixed_speech, local_tts_worker
from pilot.system.local_tts_worker import LocalTTSWorkerManager, SpeechStore, split_for_speech

GATHERED = set(fixed_speech.gather())


def pieces(text: str) -> list[str]:
    return split_for_speech(text)


@pytest.mark.parametrize(
    "sentence",
    [
        say_back.DROPPED,  # the say-back's "Okay, I've left it."
        *handover.MEDIA_ACKS.values(),  # acknowledgements
        *(line for lines in handover.STILL.values() for line in lines),  # the holding phrases
        handover.LOCAL_SLOW,
        layer.UNCLEAR_OFFLINE,
        "Working on it.",  # a code answer written as a plain return
        # The personal layer's and Live's offers: getting to know, the day start, the timeline, the wrap-up's fixed half
        "Would you like to tell me a little about yourself? Three short questions, and you can skip any of them.",
        "You start work about now. Want to hear your day?",
        "Want me to wrap up the day?",
    ],
)
def test_the_named_fixed_sentences_are_gathered(sentence):
    assert set(pieces(sentence)) <= GATHERED, sentence


def _replies(tree: ast.AST) -> list[str]:
    """Sentence-shaped literals a spoken-reply file returns, assigns to a module constant, or passes as text=."""
    found = []
    for node in ast.walk(tree):
        values = []
        if (
            isinstance(node, ast.Return)
            and node.value is not None
            or isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
            or isinstance(node, ast.keyword)
            and node.arg in {"text", "message", "say", "sentence"}
        ):
            values = [node.value]
        for value in values:
            for part in ast.walk(value):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    if fixed_speech.spoken_shape(part.value.strip()):
                        found.append(part.value.strip())
    return found


def test_no_fixed_reply_in_the_spoken_layers_is_missed():
    """Every sentence a spoken layer returns, keeps as a constant or hands over as text is in the gathered set. A
    filter in the gatherer that drops one of them fails here."""
    missed = []
    for path in fixed_speech.source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for sentence in _replies(tree):
            for piece in pieces(sentence):
                if piece not in GATHERED and fixed_speech.spoken_shape(piece):
                    missed.append(f"{path.name}: {piece}")
    assert missed == []


def test_nothing_with_a_changing_part_is_gathered():
    source = """
def reply(name, n):
    logger.info("Started the timer.")
    a = f"It's {n} past ten."
    b = "Opening {}.".format(name)
    c = "Volume is at %d." % n
    d = f"It's nearly {n}, when you stop work. Want me to wrap up the day?"
    raise ValueError("That did not work.")
    return "Done."
"""
    assert sorted(fixed_speech.sentences_in(source)) == ["Done.", "Want me to wrap up the day?"]
    assert not any("{" in piece or "%" in piece for piece in GATHERED)


def test_prompts_and_bench_files_are_not_read():
    files = {path.relative_to(fixed_speech.PILOT).as_posix() for path in fixed_speech.source_files()}
    assert "conversation/style.py" not in files and "conversation/bench.py" not in files
    assert "conversation/say_back.py" in files and "personal/offers.py" in files


# ── the store ──


def wav(path: Path, data: bytes = b"RIFF-fixed") -> Path:
    path.write_bytes(data)
    return path


def test_fixed_audio_is_kept_found_only_by_its_exact_text_and_never_evicted_by_heads(tmp_path):
    store = SpeechStore(tmp_path / "voice_heads")
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    assert store.add_fixed("Next track.", sig, wav(tmp_path / "a.wav"))
    assert store.has("Next track.", sig) and not store.has("Next track", sig)
    assert not store.has("Next track.", SpeechStore.signature("kokoro_tts", "am_michael"))
    for n in range(local_tts_worker.HEAD_CACHE_ENTRIES + 5):  # many heads said twice
        made = wav(tmp_path / f"h{n}.wav")
        store.said(f"Head {n},", sig, made)
        store.said(f"Head {n},", sig, made)
    assert SpeechStore(tmp_path / "voice_heads").lookup("Next track.", sig) is not None  # kept, and across a restart
    store.keep_only(SpeechStore.signature("kokoro_tts", "am_michael"))  # another voice: everything goes
    assert not store.has("Next track.", sig)


def test_the_store_stops_taking_fixed_audio_when_full(tmp_path, monkeypatch):
    monkeypatch.setattr(local_tts_worker, "FIXED_SPEECH_BYTES", 25)
    store = SpeechStore(tmp_path / "voice_heads")
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    assert store.add_fixed("Done.", sig, wav(tmp_path / "a.wav", b"x" * 20))
    assert not store.add_fixed("On it.", sig, wav(tmp_path / "b.wav", b"x" * 20))
    assert store.has("Done.", sig) and not store.has("On it.", sig)


class Worker:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.returncode = None
        self.stdin = self.stdout = self
        self.closed = False
        self._payload = b""

    def write(self, payload):
        self._payload = payload

    def flush(self):
        return None

    def close(self):
        self.closed = True

    def readline(self):
        request = json.loads(self._payload)
        if request.get("op") != "load":
            text = Path(request["text_file"]).read_text(encoding="utf-8")
            self.texts.append(text)
            Path(request["output_file"]).write_bytes(f"RIFF-{text}".encode())
        return b'{"status":"ok"}\n'

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def manager_with(tmp_path, monkeypatch) -> tuple[LocalTTSWorkerManager, Worker]:
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)  # the quiet stretch: tests_make_ahead_bound
    worker = Worker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    return manager, worker


@pytest.mark.asyncio
async def test_made_ahead_only_while_idle_and_quiet_then_played_without_synthesis(tmp_path, monkeypatch):
    manager, worker = manager_with(tmp_path, monkeypatch)
    rests: list[float] = []

    async def sleep(seconds):
        rests.append(seconds)
        await asyncio.sleep(0)

    manager._sleep = sleep
    talking = {"now": True}
    await manager.ensure_ready("kokoro_tts", "af_heart")
    worker.texts.clear()
    making = asyncio.ensure_future(
        manager.make_ahead(
            "kokoro_tts", "af_heart", ["Next track.", "Still on the plan."], quiet=lambda: not talking["now"]
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
    assert worker.texts == []  # the person is talking to Heliox: nothing is made
    talking["now"] = False
    assert await making == 2
    assert worker.texts == ["Next track.", "Still on the plan."] and rests  # one at a time, resting between
    worker.texts.clear()

    played: list[bytes] = []

    async def play(path: Path) -> None:
        played.append(path.read_bytes())

    await manager.speak("kokoro_tts", "Next track.", "af_heart", None, play)
    await manager.speak("kokoro_tts", "Okay. Still on the plan.", "af_heart", None, play)
    assert worker.texts == ["Okay."]  # only the piece not kept was synthesised
    assert played == [b"RIFF-Next track.", b"RIFF-Okay.", b"RIFF-Still on the plan."]
    assert await manager.make_ahead("kokoro_tts", "af_heart", ["Next track."]) == 0  # already kept
    await manager.shutdown()


@pytest.mark.asyncio
async def test_nothing_is_made_while_a_reply_is_being_spoken(tmp_path, monkeypatch):
    manager, worker = manager_with(tmp_path, monkeypatch)
    await manager.ensure_ready("kokoro_tts", "af_heart")
    worker.texts.clear()
    manager._busy = 1  # a speak() under way
    waited: list[float] = []

    async def sleep(seconds):
        waited.append(seconds)
        if len(waited) > 3:
            manager._busy = 0
        await asyncio.sleep(0)

    manager._sleep = sleep
    assert await manager.make_ahead("kokoro_tts", "af_heart", ["Done."]) == 1
    assert len(waited) >= 4  # it waited for the reply before making anything
    await manager.shutdown()


def test_what_is_said_most_is_made_first():
    order = list(fixed_speech.gather())
    first = set(order[:64])  # about the first four minutes of making ahead at a rest of five
    wanted = [*handover.MEDIA_ACKS.values(), *(line for lines in handover.STILL.values() for line in lines)]
    wanted += [say_back.DROPPED, "Working on it."]
    assert all(set(pieces(sentence)) <= first for sentence in wanted)
    offer = order.index("Want to hear your day?")
    rest = order.index(
        next(
            piece
            for piece in order
            if piece
            in set(
                fixed_speech.sentences_in(
                    (fixed_speech.PILOT / "personal" / "spoken_settings.py").read_text(encoding="utf-8")
                )
            )
        )
    )
    assert offer < rest  # the offers before the files that are not in FIRST


@pytest.mark.asyncio
async def test_nothing_is_made_on_battery_and_it_goes_on_once_plugged_in(tmp_path, monkeypatch):
    manager, worker = manager_with(tmp_path, monkeypatch)
    await manager.ensure_ready("kokoro_tts", "af_heart")
    worker.texts.clear()
    power = {"battery": True}
    polls: list[float] = []

    async def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 5:
            power["battery"] = False  # plugged in
        await asyncio.sleep(0)

    manager._sleep = sleep
    made = await manager.make_ahead("kokoro_tts", "af_heart", ["Done."], battery=lambda: power["battery"])
    assert made == 1 and len(polls) >= 5 and worker.texts == ["Done."]
    await manager.shutdown()


def test_on_battery_reads_the_power_state(monkeypatch):
    import types

    for plugged, expected in ((False, True), (True, False), (None, False)):
        fake = types.SimpleNamespace(
            sensors_battery=lambda plugged=plugged: types.SimpleNamespace(power_plugged=plugged)
        )
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
        assert local_tts_worker.on_battery() is expected
    monkeypatch.setitem(__import__("sys").modules, "psutil", types.SimpleNamespace(sensors_battery=lambda: None))
    assert local_tts_worker.on_battery() is False  # a desktop: no battery
