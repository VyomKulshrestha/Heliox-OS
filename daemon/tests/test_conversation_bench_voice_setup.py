"""The Conversation Bench sets the voice up as the daemon does, and every run says what it measured.

Until 20 November the bench built its server without start(), so the daemon's voice set-up never ran: no store of
audio made ahead (the acknowledgement hum, said only from that store, could never play), no device choice through the
GPU arbiter, and the idle voice released by the clock alone. These tests build the harness the default way with fakes
for the listener and the voice worker (no model is loaded), and check the report's first lines and the per-turn hum.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from benchmarks.conversation_bench import driver
from benchmarks.conversation_bench import timeline as tl
from benchmarks.conversation_bench.models import KINDS
from pilot import config as pilot_config
from pilot.system import gpu_arbiter
from pilot.system import local_tts_worker as tts_worker

MANAGER_FIELDS = ("_spawn", "head_cache", "keep_while", "choose_device", "on_gpu_fault", "lease_wants", "free_memory",
                  "light_reason", "on_status", "keep_loaded_seconds", "device")  # fmt: skip


def _keep_global_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything a harness build changes in this process, put back after the test."""
    manager = tts_worker.manager()
    for name in MANAGER_FIELDS:
        monkeypatch.setattr(manager, name, getattr(manager, name))
    monkeypatch.setattr(tts_worker, "LOAD_TIMEOUT_SECONDS", tts_worker.LOAD_TIMEOUT_SECONDS)
    shared = gpu_arbiter.arbiter()
    for name in ("max_temperature_c", "headroom_bytes", "fault_limit", "fault_window_seconds",
                 "speech_despite_driver_errors"):  # fmt: skip
        monkeypatch.setattr(shared, name, getattr(shared, name))
    import pilot.system.voice as voice_module

    for name in ("_play_local_tts_file", "warm_output_line", "_tts_windows", "_tts_macos", "_tts_linux",
                 "_record_audio"):  # fmt: skip
        if hasattr(voice_module, name):
            monkeypatch.setattr(voice_module, name, getattr(voice_module, name))
    for name in ("TMP", "TEMP", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # every card hidden, as on the runner
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)


class _Recorder:
    frame_size = 512
    user_is_speaking = False
    last_endpoint_reason = ""
    last_endpoint_ms = 0.0
    last_turn_probability = None

    def _publish_audio_block(self, _block: Any) -> None:
        return None


class _FakeListener:
    """Stands in for ContinuousVoiceListener: no microphone, no recogniser, no model."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._recorder = _Recorder()
        self.transcripts_received = 0
        self.user_is_speaking = False
        self._command_tasks: set[Any] = set()
        self.is_running = True

    def note_interruption(self, *_a: Any, **_k: Any) -> None:
        return None

    async def start(self) -> str:
        return "listening (fake)"

    async def stop(self, *_a: Any, **_k: Any) -> None:
        self.is_running = False


async def _build_default_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[driver.Harness, list[str]]:
    import pilot.server as server_module
    import pilot.system.voice as voice_module

    _keep_global_state(monkeypatch)
    spawned: list[Any] = []
    monkeypatch.setattr(tts_worker.manager(), "_spawn", lambda *args: spawned.append(args))
    monkeypatch.setattr(voice_module, "ContinuousVoiceListener", _FakeListener)
    prewarmed: list[str] = []
    monkeypatch.setattr(server_module.PilotServer, "_prewarm_voice", lambda self, **_k: prewarmed.append("prewarm"))
    harness = driver.Harness(audio_root=str(tmp_path), data_dir=pilot_config._TEST_ROOT, model=KINDS["cloud"],
                             wait_until_ready=False)  # fmt: skip
    await harness.build()
    assert spawned == []  # no voice worker was started by the test
    return harness, prewarmed


async def _close(harness: driver.Harness) -> None:
    if harness._vad_watch is not None:
        harness._vad_watch.cancel()
    await harness.server._live.stop()
    await tts_worker.manager().shutdown()


def test_a_harness_built_the_default_way_has_the_daemons_store_device_rule_and_release_rule(tmp_path, monkeypatch):
    async def go() -> None:
        harness, prewarmed = await _build_default_harness(tmp_path, monkeypatch)
        try:
            manager = tts_worker.manager()
            # The store the hum is said from, wired as the conversation layer reads it.
            assert isinstance(manager.head_cache, tts_worker.SpeechStore)
            # The release rule the daemon uses by default (voice.tts_release = "memory").
            assert harness.server.config.voice.tts_release == "memory"
            assert manager.keep_while is not None
            # The device chosen through the GPU arbiter, with the daemon's limits.
            assert manager.choose_device is gpu_arbiter.voice_device
            assert harness.server.config.voice.tts_device == "auto"  # no longer forced to "cpu" by the bench
            # As the server does once its listener runs: the voice loads and the fixed sentences are made ahead.
            assert prewarmed == ["prewarm"]
            lines = harness.setup["lines"]
            assert lines[0].startswith("SET-UP")
            assert "the daemon's voice set-up did not run" not in "\n".join(lines)
        finally:
            await _close(harness)

    asyncio.run(go())


def test_the_conversation_layers_hum_check_is_put_on_the_timeline(tmp_path, monkeypatch):
    async def go() -> None:
        harness, _ = await _build_default_harness(tmp_path, monkeypatch)
        try:
            from pilot.conversation.adapters import layer_for

            assert layer_for(harness.server)._ready_made("Mm.") is False  # nothing made ahead in a fresh store
            checks = tl.all_of(harness.log.events, driver.HUM_CHECK)
            assert [(c.detail, c.data["ready"]) for c in checks] == [("Mm.", False)]
        finally:
            await _close(harness)

    asyncio.run(go())


def _setup_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **voice: Any) -> driver.Harness:
    """A harness with a server stand-in that holds only settings, and the voice manager set by hand."""
    _keep_global_state(monkeypatch)
    settings = pilot_config.PilotConfig()
    for name, value in voice.items():
        setattr(settings.voice, name, value)
    harness = driver.Harness(audio_root=str(tmp_path), data_dir=tmp_path, model=KINDS["cloud"])
    harness.server = argparse.Namespace(config=settings)
    return harness


def _wire_like_the_daemon(monkeypatch: pytest.MonkeyPatch, store: tts_worker.SpeechStore) -> None:
    manager = tts_worker.manager()
    monkeypatch.setattr(manager, "choose_device", gpu_arbiter.voice_device)
    monkeypatch.setattr(manager, "keep_while", lambda: True)
    monkeypatch.setattr(manager, "head_cache", store)
    monkeypatch.setattr(manager, "effective_device", "cpu")
    monkeypatch.setattr(manager, "device_reason", "On the processor: CUDA_VISIBLE_DEVICES hides every graphics card.")
    monkeypatch.setattr(manager, "keep_loaded_seconds", 300.0)


def test_the_first_lines_say_the_device_the_store_with_the_hums_and_the_release_rule(tmp_path, monkeypatch):
    harness = _setup_harness(monkeypatch, tmp_path)
    store = tts_worker.SpeechStore(tmp_path / "voice_heads")
    _wire_like_the_daemon(monkeypatch, store)
    audio = tmp_path / "mm.wav"
    audio.write_bytes(b"RIFF")
    store.add_fixed("Mm.", tts_worker.SpeechStore.signature("kokoro_tts", "af_heart"), audio)
    harness.made_ahead_waited = {"limit_s": 150.0, "waited_s": 41.0, "ready": False}

    lines = harness.voice_setup()["lines"]

    assert lines[0] == "SET-UP: the daemon's voice set-up and default settings."
    assert lines[1] == (
        "voice: kokoro_tts af_heart on cpu: On the processor: CUDA_VISIBLE_DEVICES hides every graphics card."
    )
    assert lines[2].startswith("made-ahead store: 1 fixed pieces and 0 kept heads for this voice; hums: Mm. in, ")
    assert "Hmm. missing" in lines[2] and "not there after 41 s: no turn of this run can say a hum" in lines[2]
    assert lines[3].startswith("release rule: after 300 s idle, kept while the computer has the memory to spare")


def test_a_run_without_the_daemons_set_up_or_with_other_settings_says_so_first(tmp_path, monkeypatch):
    harness = _setup_harness(monkeypatch, tmp_path, tts_device="cpu")
    manager = tts_worker.manager()
    # What the bench had before 20 November: nothing wired.
    monkeypatch.setattr(manager, "choose_device", None)
    monkeypatch.setattr(manager, "keep_while", None)
    monkeypatch.setattr(manager, "head_cache", None)
    monkeypatch.setattr(manager, "effective_device", None)
    monkeypatch.setattr(manager, "keep_loaded_seconds", 300.0)
    monkeypatch.setattr(manager, "_failed_until", None)
    harness.cards_hidden_by = "bench"

    setup = harness.voice_setup()
    lines = setup["lines"]

    assert lines[0].startswith("SET-UP DIFFERS from the daemon's default")
    differs = [line for line in lines if line.startswith("  differs: ")]
    assert lines[1 : 1 + len(differs)] == differs  # every difference before the figures' own lines
    assert "  differs: voice.tts_device 'auto' -> 'cpu'" in differs
    assert "  differs: every graphics card hidden by the bench (CUDA_VISIBLE_DEVICES set empty)" in differs
    assert any("the daemon's voice set-up did not run" in line and "no store of audio made ahead" in line
               for line in differs)  # fmt: skip
    assert any(line.startswith("made-ahead store: none") for line in lines)
    assert any(line.startswith("release rule: by the clock alone, 300 s") for line in lines)


def test_a_path_setting_is_named_without_its_value():
    run, default = pilot_config.PilotConfig(), pilot_config.PilotConfig()
    run.voice.parakeet_model_dir = "C:/somewhere/models/parakeet"
    run.conversation.acknowledge = False

    assert driver.settings_differences(run, default) == [
        "voice.parakeet_model_dir '' -> (a different value)",
        "conversation.acknowledge True -> False",
    ]


def _turn(*events: tuple[float, str, str, dict[str, Any]]) -> list[tl.Event]:
    return [tl.Event(t, kind, detail, data) for t, kind, detail, data in events]


def test_each_turn_says_whether_a_hum_was_said_and_from_where():
    said = _turn(
        (1.0, tl.LAST_WORD, "q", {"audio_t": 1.0}),
        (1.2, driver.HUM_CHECK, "Mm.", {"ready": True}),
        (1.5, tl.SPEAKS, "Mm.", {"ack": True}),
        (1.55, tl.AUDIO_STARTS, "", {"audio_t": 1.55}),
        (1.8, tl.VOICE_REPLY, "", {"ack": True, "cut": True, "voice": "store", "branch": "store"}),
        (1.8, tl.CUT_OFF, "", {"ack": True}),
        (1.9, tl.SPEAKS, "Sure.", {}),
    )
    found = driver.acknowledgement(said)
    assert found == {"said": "Mm.", "voice": "store", "cut": True, "after_last_word_s": 0.55}
    assert driver.acknowledgement_text(found) == "hum 'Mm.' from store at 0.55 s after the last word, cut by the answer"

    not_made = _turn((1.0, tl.LAST_WORD, "q", {}), (1.2, driver.HUM_CHECK, "Mm.", {"ready": False}))
    assert driver.acknowledgement(not_made) == {"said": None, "why": "the hum's audio was not made ahead"}
    never_asked = _turn((1.0, tl.LAST_WORD, "q", {}), (1.5, tl.SPEAKS, "Sure.", {}))
    assert driver.acknowledgement(never_asked)["why"].startswith("not asked")


def test_a_hum_cut_by_the_answer_is_not_a_stop():
    events = _turn(
        (0.5, tl.FIRST_WORD, "q", {}),
        (1.0, tl.LAST_WORD, "q", {}),
        (1.5, tl.SPEAKS, "Mm.", {"ack": True}),
        (1.55, tl.AUDIO_STARTS, "", {}),
        (1.8, tl.CUT_OFF, "", {"ack": True}),
        (1.9, tl.SPEAKS, "Sure.", {}),
        (2.0, tl.AUDIO_STARTS, "", {}),
        (2.4, tl.DONE, "", {}),
    )
    measured = tl.measure(events)
    assert measured.stop_time is None
    assert measured.reply_start == pytest.approx(0.55)  # the hum is Heliox's first sound


def test_voice_latency_keeps_a_hum_out_of_the_answers_first_sentence_and_route():
    from benchmarks.conversation_bench import voice_latency

    events = _turn(
        (1.0, tl.LAST_WORD, "q", {"audio_t": 1.0}),
        (1.6, tl.DISPATCH, "Hey Heliox, why is the sky blue?", {"audio_t": 1.6}),
        (1.7, tl.SPEAKS, "Mm.", {"ack": True, "audio_t": 1.7}),
        (1.72, tl.AUDIO_STARTS, "", {"audio_t": 1.72}),
        (1.75, tl.VOICE_REPLY, "", {"ack": True, "voice": "store", "branch": "store"}),
        (2.2, voice_latency.MODEL_FIRST, "", {"audio_t": 2.2}),
        (2.4, tl.SPEAKS, "Sure.", {"audio_t": 2.4}),
        (2.9, tl.AUDIO_STARTS, "", {"audio_t": 2.9}),
        (2.95, tl.VOICE_REPLY, "", {"voice": "local", "branch": "ready"}),
    )
    found = voice_latency.stages(events)

    assert found["first sound"]["audio"] == pytest.approx(0.72)  # the hum is what the person hears first
    assert found["first sentence"]["audio"] == pytest.approx(1.4)  # the answer's, not the hum
    assert found["answer sound"]["audio"] == pytest.approx(1.9)
    assert found["route"] == "model answer"
    assert found["ack"]["said"] == "Mm." and found["ack"]["voice"] == "store"
    assert voice_latency.reply_voice(events)["spoke_with"] == "local"  # the reply's voice, not the hum's
    assert voice_latency.hums_line([{"ack": found["ack"]}, {"ack": {"said": None}}]) == (
        "hums said: 1 of 2 turns (from store: 1)"
    )


def test_the_text_only_replay_starts_no_voice_at_all(monkeypatch):
    """text_replay models the voice: no local voice worker, no voice set-up, nothing made ahead."""
    import pilot.server as server_module
    from benchmarks.conversation_bench import text_replay

    _keep_global_state(monkeypatch)
    spawned: list[Any] = []
    set_up: list[str] = []
    monkeypatch.setattr(tts_worker.manager(), "_spawn", lambda *args: spawned.append(args))
    monkeypatch.setattr(server_module.PilotServer, "_set_up_voice", lambda self: set_up.append("set up"))
    monkeypatch.setattr(text_replay, "free_gb", lambda: None)
    monkeypatch.setattr("pilot.real_folders.apply_owner_settings", lambda *a, **k: [])  # no person's settings read
    args = argparse.Namespace(root=str(pilot_config._TEST_ROOT), turns=1, compare="", model="stub", paid=False,
                              max_calls=1, start_free_gb=0.0, min_free_gb=0.0, max_usd=0.0, warm_lead=0.0,
                              handover_s=0.6, wait_quiet=0.0, threshold=100.0, out=None)  # fmt: skip

    assert asyncio.run(text_replay._main(args)) == 0

    assert spawned == [] and set_up == []
    assert tts_worker.manager().process is None
