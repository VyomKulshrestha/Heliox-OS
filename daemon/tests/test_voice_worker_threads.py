"""The voice worker makes each piece on its own thread, with no event loop started per request, and gives the model
four threads (fewer on a smaller machine). Fake engine and a fake torch: no model."""

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

    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
    assert made == [("Volume is at 30.", "af_heart", str(tmp_path / "o.wav"), True, False)]
    assert len(fake_torch) == 1
