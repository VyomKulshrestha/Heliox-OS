"""A voice file is fetched only when the person presses the button that offers it, on every path; a missing voice is
the voice's failure, never the engine's (Kokoro included); Settings, Voice offers the download.

Every path that could fetch a voice file is walked here: the face picker and Settings (behind their button), a
stored config at start, and everything inside the voice worker (the standby's load, the made-ahead maker, Pocket's and
Kokoro's own lazy fetch), which runs with Hugging Face fetching off once the engine's model is on the computer.
Fakes only: nothing is fetched, no model loads, no network or card is used."""

from __future__ import annotations

import inspect
import io
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import kokoro_tts, local_tts_worker, voice_default, voice_fetch
from pilot.system import local_tts_worker as ltw


def _no_fetch(*_args, **_kwargs):
    raise AssertionError("a voice file was fetched without the button")


# ── Inside the voice worker: fetching is off once the model is here ──


def test_the_worker_turns_fetching_off_once_the_engine_s_model_is_here(monkeypatch):
    fake_hub = types.SimpleNamespace(constants=types.SimpleNamespace(HF_HUB_OFFLINE=False))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_hub.constants)
    here: dict[str, str] = {}
    assert ltw.voices_offline("pocket_tts", here, lambda: frozenset({"pocket_tts"})) is True
    assert here["HF_HUB_OFFLINE"] == "1" and fake_hub.constants.HF_HUB_OFFLINE is True
    first_run: dict[str, str] = {}
    assert ltw.voices_offline("kokoro_tts", first_run, lambda: frozenset({"pocket_tts"})) is False
    assert "HF_HUB_OFFLINE" not in first_run  # a first run fetches its model and default voice, as it chose to
    assert ltw.voices_offline("kokoro_tts", {"HF_HUB_OFFLINE": "1"}, lambda: frozenset()) is True


def test_fetching_is_turned_off_before_anything_loads_in_the_worker():
    """The standby's load, the made-ahead maker and every reply run in this loop, after this line."""
    source = inspect.getsource(ltw._serve)
    assert (
        source.index("voices_offline(engine)") < source.index("kokoro_tts.configure") < source.index("requests.next()")
    )


def test_kokoro_never_turns_fetching_back_on_once_the_worker_turned_it_off(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(kokoro_tts, "_offline_by_us", False)
    assert kokoro_tts.use_cached_weights_offline("am_michael") is True
    kokoro_tts._allow_fetching_missing_voice("am_michael")  # a voice not on disk: before, this let Kokoro fetch it
    assert local_tts_worker.os.environ["HF_HUB_OFFLINE"] == "1"


def test_the_worker_says_when_a_voice_s_own_file_is_missing(monkeypatch, capsys):
    def generate_here(text, voice_name, timing=False):
        raise RuntimeError("LocalEntryNotFoundError: Cannot find the requested files in the disk cache")

    monkeypatch.setattr(ltw, "_engine_module", lambda engine: SimpleNamespace(generate_here=generate_here))
    monkeypatch.setattr(ltw, "_use_voice_threads", lambda *_: None)
    monkeypatch.setattr(ltw, "voices_offline", lambda engine: True)
    monkeypatch.setattr(ltw, "voice_installed", lambda engine, name: name != "am_michael")
    request = {"text": "Okay.", "voice": "am_michael", "reply": "samples"}
    monkeypatch.setattr(ltw.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    assert ltw._serve("kokoro_tts") == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "error" and answer["voice_missing"] is True


# ── The manager: a missing or silent piece on Kokoro never marks Kokoro unavailable ──


class Worker:
    def __init__(self, answer: dict) -> None:
        self.answer, self.returncode, self._last = answer, None, b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=lambda: None, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, close=lambda: None)

    def _write(self, data: bytes) -> None:
        self._last = data

    def _readline(self) -> bytes:
        request = json.loads(self._last)
        if request.get("op") == "load":
            return b'{"status": "ok", "op": "load"}\n'
        return (json.dumps(self.answer) + "\n").encode()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ({"status": "error", "error": "RuntimeError: Kokoro produced no audio"}, ltw.LocalTTSNoAudio),
        ({"status": "error", "error": "LocalEntryNotFoundError: voices/am_michael.pt", "voice_missing": True},
         ltw.LocalTTSVoiceMissing),
    ],
)  # fmt: skip
async def test_a_kokoro_piece_that_is_silent_or_whose_voice_is_missing_leaves_kokoro_working(tmp_path, answer, kind):
    worker = Worker(answer)
    manager = ltw.LocalTTSWorkerManager(spawn=lambda *_: worker, device="cpu")
    manager.environ = {}
    await manager.ensure_ready("kokoro_tts", "af_heart")
    with pytest.raises(kind):
        await manager.synthesize("kokoro_tts", "Okay.", "am_michael", tmp_path / "o.wav", tmp_path)
    assert manager.failing is False and manager.running
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_reply_in_a_missing_voice_is_said_with_the_voice_that_worked(monkeypatch):
    from pilot.system import voice

    manager = ltw.LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.voice_spoke("kokoro_tts", "af_heart")
    monkeypatch.setattr(ltw, "_manager", manager)
    said: list[str] = []

    async def run(engine, text, voice_name, output_file, *, wait_for_load=False):
        said.append(voice_name)
        if voice_name == "am_michael":
            raise ltw.LocalTTSVoiceMissing(engine, voice_name)

    monkeypatch.setattr(voice, "_run_local_tts_worker", run)
    await voice._run_local_voice("kokoro_tts", "Okay.", "am_michael", None)
    assert said == ["am_michael", "af_heart"] and manager.put_aside[("kokoro_tts", "am_michael")] == "missing"


# ── The daemon: Settings, Voice, a stored config, and the button ──


def _server(monkeypatch, *, engine="pocket_tts", voice="michael", here=frozenset({"alba"})):
    config = PilotConfig()
    config.voice.language = "en"
    config.voice.tts_engine, config.voice.tts_voice = engine, voice
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._voice_download_check = lambda engine_: (True, "")
    manager = ltw.LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.voice_installed = lambda engine_, name: name in here
    manager.voice_spoke("pocket_tts", "alba")
    monkeypatch.setattr(ltw, "_manager", manager)
    monkeypatch.setattr(voice_fetch, "fetch", _no_fetch)
    monkeypatch.setattr(voice_default, "voice_models_on_disk", lambda cache=None: frozenset({"pocket_tts"}))
    return server, manager


@pytest.mark.asyncio
async def test_a_stored_voice_that_is_not_here_is_shown_once_in_settings_with_its_button(monkeypatch):
    server, manager = _server(monkeypatch)
    listed = await server._voices_aside()
    assert [(item["voice"], item["reason"], item["download_bytes"], item["download_ok"]) for item in listed] == [
        ("michael", "missing", 7_128_888, True)
    ]
    assert listed[0]["say"] == (
        "The voice michael is not on this computer yet; it needs a download of about 7 MB. I'll keep speaking "
        "with alba."
    )
    assert ltw.voice_for_reply("pocket_tts", "michael") == ("pocket_tts", "alba")  # the voice that worked speaks


@pytest.mark.asyncio
async def test_settings_download_runs_only_on_the_button_then_the_test_sentence(monkeypatch):
    server, manager = _server(monkeypatch)
    asked = await server._handle_voice_download({"engine": "pocket_tts", "voice": "michael", "apply": ""}, None)
    assert asked["installed"] is False and asked["download_bytes"] == 7_128_888  # asking fetches nothing
    fetched: list[list[voice_fetch.HubFile]] = []
    server._voice_fetcher = fetched.append
    retried: list[tuple[str, str]] = []
    monkeypatch.setattr(ltw, "retry_voice", lambda engine, name: retried.append((engine, name)) or "prove")
    started = await server._handle_voice_download(
        {"engine": "pocket_tts", "voice": "michael", "apply": "download"}, None
    )
    assert started["status"] == "ok" and started["downloading"] is True
    await server._voice_download_task
    assert [f.path for f in fetched[0]] == ["languages/english/embeddings/michael.safetensors"]
    assert fetched[0][0].revision == "4e1e0a3e611c51c0b4ed8174fc10f32a54644303"  # where pocket-tts reads it
    assert retried == [("pocket_tts", "michael")]  # then the voice says its test sentence before it speaks
    statuses = [call.args[1]["status"] for call in server._broadcast_notification.await_args_list]
    assert statuses == ["voice_downloaded"]


@pytest.mark.asyncio
async def test_settings_download_is_refused_offline_and_for_a_voice_not_on_the_list(monkeypatch):
    server, _manager = _server(monkeypatch)
    server._voice_download_check = lambda engine_: (False, "No connection to huggingface.co right now.")
    refused = await server._handle_voice_download(
        {"engine": "pocket_tts", "voice": "michael", "apply": "download"}, None
    )
    assert refused["status"] == "error" and refused["message"] == "No connection to huggingface.co right now."
    for engine, name in (("pocket_tts", "cosette"), ("pocket_tts", "lola"), ("kokoro_tts", "alba"), ("other", "x")):
        bad = await server._handle_voice_download({"engine": engine, "voice": name, "apply": "download"}, None)
        assert bad["status"] == "error"


@pytest.mark.asyncio
async def test_a_stored_config_at_start_never_fetches(monkeypatch):
    server, _manager = _server(monkeypatch)
    server._choose_first_voice(machine=lambda device: "processor", models_on_disk=lambda: frozenset({"pocket_tts"}),
                               ever_loaded=lambda: True)  # fmt: skip
    ltw.voice_for_reply("pocket_tts", "michael")
    await server._voices_aside()
    # voice_fetch.fetch raises if called: reaching here means no path above fetched a voice file


def test_a_voice_download_is_retried_with_its_test_sentence(monkeypatch):
    manager = ltw.LocalTTSWorkerManager(spawn=lambda *_: None)
    manager.process, manager.engine, manager.loaded = SimpleNamespace(poll=lambda: None), "pocket_tts", True
    manager.put_aside[("pocket_tts", "michael")] = "missing"
    monkeypatch.setattr(ltw, "_manager", manager)
    monkeypatch.setattr(ltw, "_spawn_task", lambda coroutine: coroutine.close())
    assert ltw.retry_voice("pocket_tts", "michael") == "prove"
    assert ("pocket_tts", "michael") not in manager.put_aside and ("pocket_tts", "michael") in manager.proving
