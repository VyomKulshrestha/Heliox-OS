"""A voice that makes no sound, or whose file is not on the computer, never takes the voice down and never speaks.

On 27 November, with Pocket speaking, the man's face brought michael; michael's "Hmm.", made ahead right after the
switch, came out with no sound, the worker answered "Pocket TTS produced no audio", and the manager marked the whole
engine unavailable: the next three replies came in the system voice for about 92 s. Now that is the piece's failure:
the hum is not kept, the engine stays, a voice that makes no sound for a reply is put aside by name and the reply is
said with the last voice of the engine that made sound; a new voice says a test sentence before it speaks a reply,
and a voice whose own file is not here is never switched to while another voice can speak.

Made-up workers, managers and stores only: no model, no audio device, no card, no network; orders are counted."""

from __future__ import annotations

import importlib.metadata
import io
import json
import re
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import face_voice, local_tts_worker, voice, voice_default, voice_fetch
from pilot.system import local_tts_worker as ltw
from pilot.system.local_tts_worker import LocalTTSNoAudio, LocalTTSWorkerManager, Piece

LOUD = np.full(240, 0.3, dtype=np.float32)


# ── The worker and the manager: no sound is the piece's failure, not the engine's ──


def test_the_worker_says_when_a_text_came_out_with_no_sound(monkeypatch, capsys):
    def generate_here(text, voice_name, timing=False):
        raise RuntimeError("Pocket TTS produced no audio")

    monkeypatch.setattr(ltw, "_engine_module", lambda engine: SimpleNamespace(generate_here=generate_here))
    monkeypatch.setattr(ltw, "_use_voice_threads", lambda *_: None)
    request = {"text": "Hmm.", "voice": "michael", "reply": "samples"}
    monkeypatch.setattr(ltw.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    assert ltw._serve("pocket_tts") == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "error" and answer["no_audio"] is True


class Worker:
    """A stand-in worker process: loads, then answers every piece with ``answer`` (counted)."""

    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.asked: list[dict] = []
        self.returncode = None
        self._last = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=lambda: None, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, close=lambda: None)

    def _write(self, data: bytes) -> None:
        self._last = data

    def _readline(self) -> bytes:
        request = json.loads(self._last)
        self.asked.append(request)
        if request.get("op") == "load":
            return b'{"status": "ok", "op": "load"}\n'
        return (json.dumps(self.answer) + "\n").encode()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_a_piece_with_no_sound_never_marks_the_engine_unavailable(tmp_path):
    worker = Worker({"status": "error", "error": "RuntimeError: Pocket TTS produced no audio", "no_audio": True})
    manager = LocalTTSWorkerManager(spawn=lambda *_: worker)
    manager.environ = {}
    await manager.ensure_ready("pocket_tts", "alba")
    with pytest.raises(LocalTTSNoAudio) as caught:
        await manager.synthesize("pocket_tts", "Hmm.", "michael", tmp_path / "o.wav", tmp_path)
    assert caught.value.voice == "michael" and caught.value.engine == "pocket_tts"
    assert local_tts_worker.NO_AUDIO in str(caught.value)
    assert manager.failing is False and manager.running  # the engine and its worker go on
    # The same for an answer from an older worker that does not carry the flag.
    worker.answer = {"status": "error", "error": "RuntimeError: Pocket TTS produced no audio"}
    with pytest.raises(LocalTTSNoAudio):
        await manager.synthesize("pocket_tts", "Hmm.", "michael", tmp_path / "o.wav", tmp_path)
    assert manager.failing is False
    # A real failure still fails the engine, as before.
    worker.answer = {"status": "error", "error": "RuntimeError: weights could not be read"}
    with pytest.raises(ltw.LocalTTSUnavailable) as other:
        await manager.synthesize("pocket_tts", "Okay.", "alba", tmp_path / "o.wav", tmp_path)
    assert not isinstance(other.value, LocalTTSNoAudio) and manager.failing is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_hum_with_no_sound_is_noted_and_the_rest_are_still_made():
    manager = LocalTTSWorkerManager(spawn=lambda *_: None, clock=lambda: 100.0)
    manager.process, manager.engine, manager.loaded = SimpleNamespace(poll=lambda: None), "pocket_tts", True
    kept: set[str] = set()
    silent: set[str] = set()
    manager.head_cache = SimpleNamespace(
        has=lambda text, sig: text in kept,
        add_fixed=lambda text, sig, piece: kept.add(text) or True,
        is_silent=lambda text, sig: text in silent,
        mark_silent=lambda text, sig: silent.add(text),
    )

    async def make(engine, text, voice_name, path, folder, **_given):
        if text == "Hmm.":
            raise LocalTTSNoAudio(engine, voice_name)
        return Piece(text, audio=LOUD, rate=24000)

    async def never_warm():
        return False

    async def no_wait(_seconds):
        return None

    manager._make, manager.too_warm_now, manager._sleep = make, never_warm, no_wait
    made = await manager.make_ahead("pocket_tts", "michael", ["Hmm.", "Mm.", "Okay."], idle_seconds=0.0,
                                    quiet=lambda: True, battery=lambda: False, rest_factor=0.0)  # fmt: skip
    assert made == 2 and kept == {"Mm.", "Okay."} and silent == {"Hmm."}
    assert manager.failing is False and manager.put_aside == {}  # one silent hum is not a broken voice


# ── A reply in a voice that makes no sound ──


@pytest.mark.asyncio
async def test_a_reply_with_no_sound_is_said_with_the_last_voice_that_worked(monkeypatch):
    manager = LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.voice_spoke("pocket_tts", "alba")
    told: list[tuple[str, dict]] = []

    async def status(name, data):
        told.append((name, data))

    manager.on_status = status
    monkeypatch.setattr(ltw, "_manager", manager)
    said: list[tuple[str, str]] = []

    async def run(engine, text, voice_name, output_file, *, wait_for_load=False):
        said.append((voice_name, text))
        if voice_name == "michael":
            raise LocalTTSNoAudio(engine, voice_name)

    monkeypatch.setattr(voice, "_run_local_tts_worker", run)
    await voice._run_local_voice("pocket_tts", "Ten minutes.", "michael", None)
    assert said == [("michael", "Ten minutes."), ("alba", "Ten minutes.")]
    assert manager.put_aside == {("pocket_tts", "michael"): "no sound"}
    assert [name for name, _ in told] == ["voice_put_aside"]
    assert told[0][1]["message"] == "The voice michael made no sound, so Heliox keeps speaking with alba."
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "alba")  # the next replies too
    await voice._run_local_voice("pocket_tts", "Again.", "michael", None)
    assert [name for name, _ in told] == ["voice_put_aside"]  # said once


@pytest.mark.asyncio
async def test_with_no_voice_that_worked_the_system_voice_says_it(monkeypatch):
    manager = LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.voice_installed = lambda engine, name: False
    monkeypatch.setattr(ltw, "_manager", manager)

    async def run(engine, text, voice_name, output_file, *, wait_for_load=False):
        raise LocalTTSNoAudio(engine, voice_name)

    monkeypatch.setattr(voice, "_run_local_tts_worker", run)
    with pytest.raises(LocalTTSNoAudio):
        await voice._run_local_voice("pocket_tts", "Ten minutes.", "michael", None)


# ── A new voice is proved before it speaks a reply ──


def _loaded(engine="pocket_tts") -> LocalTTSWorkerManager:
    manager = LocalTTSWorkerManager(spawn=lambda *_: pytest.fail("a proof never starts a worker"), clock=lambda: 1.0)
    manager.process, manager.engine, manager.loaded = SimpleNamespace(poll=lambda: None), engine, True
    manager.voice_spoke(engine, "alba" if engine == "pocket_tts" else "af_heart")
    return manager


@pytest.mark.asyncio
async def test_a_voice_whose_file_is_missing_is_put_aside_with_its_download_size():
    manager = _loaded()
    manager.voice_installed = lambda engine, name: name != "michael"
    told: list[dict] = []

    async def status(name, data):
        told.append(data)

    manager.on_status = status
    assert await manager.prove_voice("pocket_tts", "michael") is False
    assert manager.put_aside[("pocket_tts", "michael")] == "missing"
    assert told[0]["download_bytes"] == 7_128_888 and told[0]["speaking"] == "alba"
    assert told[0]["message"] == (
        "The voice michael is not on this computer yet; it needs a download of about 7 MB. Heliox keeps speaking "
        "with alba."
    )


@pytest.mark.asyncio
async def test_a_voice_proves_itself_with_sound_or_is_put_aside():
    manager = _loaded()
    manager.voice_installed = lambda engine, name: True
    heard: list[str] = []

    async def make(engine, text, voice_name, path, folder, **given):
        assert given["start_voice"] is False and given["rank"] == ltw.PIPE_AHEAD  # behind any reply, never a cold start
        heard.append(voice_name)
        if voice_name == "silent_voice":
            return Piece(text, audio=np.zeros(240, dtype=np.float32), rate=24000)
        if voice_name == "broken_voice":
            raise LocalTTSNoAudio(engine, voice_name)
        return Piece(text, audio=LOUD, rate=24000)

    manager._make = make
    assert await manager.prove_voice("pocket_tts", "michael") is True
    assert ("pocket_tts", "michael") in manager.spoke_voices
    assert await manager.prove_voice("pocket_tts", "silent_voice") is False
    assert await manager.prove_voice("pocket_tts", "broken_voice") is False
    assert set(manager.put_aside) == {("pocket_tts", "silent_voice"), ("pocket_tts", "broken_voice")}
    assert heard == ["michael", "silent_voice", "broken_voice"] and manager.failing is False


@pytest.mark.asyncio
async def test_a_proof_waits_a_counted_while_for_the_worker_and_never_starts_one():
    manager = LocalTTSWorkerManager(spawn=lambda *_: pytest.fail("never started"))
    manager.voice_installed = lambda engine, name: True
    waits = 0

    async def sleep(_seconds):
        nonlocal waits
        waits += 1

    manager._sleep = sleep
    assert await manager.prove_voice("pocket_tts", "michael", polls=3) is False
    assert waits == 3 and manager.put_aside == {}  # not loaded in time: tried on its first reply, as before


def test_while_a_new_voice_is_proved_the_one_that_worked_speaks(monkeypatch):
    manager = _loaded()
    manager.proving.add(("pocket_tts", "michael"))
    monkeypatch.setattr(ltw, "_manager", manager)
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "alba")
    manager.voice_spoke("pocket_tts", "michael")
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "michael")


def test_a_setting_naming_a_voice_that_is_not_here_never_speaks_while_another_can(monkeypatch):
    """A config from another computer, or a voice picked before its download."""
    manager = _loaded()
    manager.voice_installed = lambda engine, name: name == "alba"
    told: list[str] = []
    manager.record_aside = (lambda original: lambda *a: told.append(a[1]) or original(*a))(manager.record_aside)
    monkeypatch.setattr(ltw, "_manager", manager)
    monkeypatch.setattr(ltw, "_spawn_task", lambda coroutine: coroutine.close())
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "alba")
    assert manager.put_aside[("pocket_tts", "michael")] == "missing" and told == ["michael"]


def test_with_nothing_else_to_speak_with_the_named_voice_is_used_as_before(monkeypatch):
    manager = LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.voice_installed = lambda engine, name: False
    monkeypatch.setattr(ltw, "_manager", manager)
    assert ltw.voice_for_reply("pocket_tts", "alba") == ("pocket_tts", "alba")
    assert manager.put_aside == {}


@pytest.mark.asyncio
async def test_switching_voices_of_one_engine_proves_the_new_one_first(monkeypatch):
    manager = _loaded()
    manager.process_device = manager.device
    proved: list[str] = []

    async def prove(engine, name, **_):
        proved.append(name)
        return True

    manager.prove_voice = prove
    monkeypatch.setattr(ltw, "_manager", manager)
    monkeypatch.setattr(ltw, "_make_ready_soon", lambda *_: None)
    assert ltw._begin_switch("pocket_tts", "michael", "alba") == "same"
    assert ("pocket_tts", "michael") in manager.proving  # alba speaks until michael has said its test sentence
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "alba")


def test_a_standby_whose_voice_fails_its_test_never_takes_over(monkeypatch):
    old = _loaded("kokoro_tts")
    old.voice_spoke("kokoro_tts", "af_heart")
    old.free_memory = lambda: 16 * 1024**3
    monkeypatch.setattr(ltw, "_manager", old)
    monkeypatch.setattr(ltw, "_spawn_task", lambda coroutine: coroutine.close())
    standby = LocalTTSWorkerManager(spawn=lambda *_: None)
    standby._start_prewarm = lambda *_: None
    assert ltw._begin_switch("pocket_tts", "michael", "af_heart", make_standby=lambda: standby) == "standby"
    standby.process, standby.engine, standby.loaded = SimpleNamespace(poll=lambda: None), "pocket_tts", True
    standby.put_aside[("pocket_tts", "michael")] = "no sound"  # its test sentence came out silent
    assert ltw.finish_handover_if_due() is False and ltw.handover() is None and ltw.manager() is old
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("kokoro_tts", "af_heart")  # the voice that worked goes on


# ── Every face's voices, on both engines: installable through the asked-for download, never switched to missing ──


@pytest.mark.parametrize("pack", sorted(face_voice.PACK_VOICES))
@pytest.mark.parametrize("engine", ["kokoro_tts", "pocket_tts"])
def test_every_face_voice_can_be_fetched_and_a_missing_one_leaves_the_old_voice(monkeypatch, pack, engine):
    named = face_voice.PACK_VOICES[pack].by_engine()[engine]
    if not named:
        pytest.skip(f"{pack} has no {engine} voice")
    files = voice_fetch.files_for(engine, named, model_on_disk=True)
    assert files and files[-1].path.endswith(
        (f"{named}.pt", f"{named}.safetensors")
    )  # what the asked-for download gets
    assert voice_fetch.download_bytes(engine, model_on_disk=True, voice=named) > 0  # and the size the question says
    if engine == "pocket_tts":
        assert voice_default.pocket_voice_allowed(named)
    manager = _loaded(engine)
    other = "af_heart" if engine == "kokoro_tts" else "alba"
    manager.voice_spoke(engine, other)
    manager.voice_installed = lambda e, name: name != named
    monkeypatch.setattr(ltw, "_manager", manager)
    monkeypatch.setattr(ltw, "_spawn_task", lambda coroutine: coroutine.close())
    if named == other:
        return  # the face's voice is the one already speaking
    assert ltw.voice_for_reply(engine, named) == (engine, other)


def test_the_pocket_voice_file_is_where_the_installed_pocket_tts_reads_it():
    try:
        source = importlib.metadata.distribution("pocket-tts").locate_file("pocket_tts/utils/utils.py")
        text = open(source, encoding="utf-8").read()  # noqa: PTH123 - a path from the package's own metadata
    except (importlib.metadata.PackageNotFoundError, OSError):
        pytest.skip("pocket-tts is not installed here")
    found = re.search(
        r"hf://([\w.-]+/[\w.-]+)/languages/\{language\}/embeddings/\{name\}\.safetensors@([0-9a-f]+)", text
    )
    assert found, "pocket-tts names its voice files another way: update speech_models.json voice_file"
    spec = voice_default.VOICES["pocket_tts"]["voice_file"]
    assert (spec["repo"], spec["revision"]) == (found.group(1), found.group(2))
    assert spec["path"] == "languages/english/embeddings/{voice}.safetensors"
