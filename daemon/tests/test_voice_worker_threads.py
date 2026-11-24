"""The voice worker makes each piece on its own thread, with no event loop started per request, and gives Kokoro four
threads and Pocket TTS one (fewer on a smaller machine). Fake engine and a fake torch: no model."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import types

import pytest

from pilot.system import local_tts_worker


@pytest.fixture
def fake_torch(monkeypatch):
    asked: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=asked.append))
    monkeypatch.setattr(local_tts_worker, "_threads_set", False)
    return asked


@pytest.mark.parametrize(("cores", "threads"), [(24, 4), (8, 4), (4, 4), (2, 2), (1, 1)])
def test_the_model_gets_four_threads_at_most_once(fake_torch, monkeypatch, cores, threads):
    monkeypatch.setattr(local_tts_worker.os, "cpu_count", lambda: cores)
    local_tts_worker._use_voice_threads()
    local_tts_worker._use_voice_threads()
    assert fake_torch == [threads]


def test_loading_the_voice_asks_for_them(fake_torch, monkeypatch):
    module = types.SimpleNamespace(prepare=lambda voice: None)
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    local_tts_worker._load("kokoro_tts", "af_heart")
    assert fake_torch == [min(4, local_tts_worker.os.cpu_count() or 4)]


@pytest.mark.parametrize(("engine", "threads"), [("kokoro_tts", 4), ("pocket_tts", 1), ("another_engine", 4)])
def test_each_engine_gets_its_own_thread_count(engine, threads):
    assert local_tts_worker.voice_threads(engine) == threads


@pytest.mark.parametrize(("engine", "cores", "threads"), [("pocket_tts", 24, 1), ("pocket_tts", 4, 1),
                                                           ("pocket_tts", 1, 1), ("kokoro_tts", 4, 4),
                                                           ("kokoro_tts", 2, 2)])  # fmt: skip
def test_loading_gives_the_engine_its_threads(fake_torch, monkeypatch, engine, cores, threads):
    module = types.SimpleNamespace(prepare=lambda voice: None)
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda name: module)
    monkeypatch.setattr(local_tts_worker.os, "cpu_count", lambda: cores)
    local_tts_worker._load(engine, "alba")
    assert fake_torch == [threads]


def test_a_pocket_stream_runs_on_one_thread(fake_torch, monkeypatch):
    """GitHub's 4-core runners streamed Pocket fastest on one thread (pocket-speed.yml): the stream asks for one."""
    module = types.SimpleNamespace(stream_here=lambda text, voice, stop: (24000, iter([[0.0, 0.1]])))
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda name: module)
    monkeypatch.setattr(local_tts_worker.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(local_tts_worker, "_send", lambda response, payload=b"": None)
    requests = types.SimpleNamespace(stop_event=lambda stream_id: threading.Event(), done=lambda stream_id: None)
    local_tts_worker._stream("pocket_tts", {"id": 1, "text": "Done.", "voice": "alba"}, requests)
    assert fake_torch == [1]


def test_a_request_is_made_on_the_worker_s_own_thread_without_an_event_loop(fake_torch, monkeypatch, tmp_path, capsys):
    made: list[tuple[str, str, str, bool, bool]] = []

    def here(text, voice, output_file):
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        made.append((text, voice, output_file, threading.current_thread() is threading.main_thread(), in_loop))

    module = types.SimpleNamespace(synthesize_to_file_here=here)
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    text_file = tmp_path / "t.txt"
    text_file.write_text("Volume is at 30.", encoding="utf-8")
    request = {"text_file": str(text_file), "voice": "af_heart", "output_file": str(tmp_path / "o.wav")}
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))

    assert local_tts_worker._serve("pocket_tts") == 0

    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "ok" and answer["took"]["read_s"] >= 0  # each step's seconds, for the trace
    assert made == [("Volume is at 30.", "af_heart", str(tmp_path / "o.wav"), True, False)]
    assert len(fake_torch) == 1
