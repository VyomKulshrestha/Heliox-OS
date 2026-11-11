"""The local model status the services strip and the bench read: same keys and words on every system."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.server import PilotServer
from pilot.system import gpu_arbiter, model_health
from pilot.system.gpu_arbiter import GpuArbiter, MachineReading
from pilot.system.local_tts_worker import LocalTTSUnavailable, LocalTTSWorkerManager

KEYS = {"name", "state", "reason", "device", "process", "last_fault"}
CUDA_FAULT = json.dumps({"status": "error", "error": "RuntimeError: CUDA error: unknown error"}).encode() + b"\n"


def _manager(**state) -> SimpleNamespace:
    base = {"running": False, "engine": None, "loaded": False, "failing": False, "effective_device": None}
    return SimpleNamespace(**{**base, "_load_task": None, **state})


def test_gnomon_states_map_to_one_vocabulary():
    entry = model_health.gnomon_entry(
        {
            "state": "unavailable",
            "reason": "Gnomon's process stopped.",
            "process": "separate",
            "device": None,
            "last_fault": {"kind": "exit", "message": "x", "at": "2026-11-12T10:00:00"},
        }  # fmt: skip
    )
    assert set(entry) == KEYS
    assert entry["state"] == "unavailable" and entry["process"] == "worker" and entry["last_fault"]["kind"] == "exit"
    mac = model_health.gnomon_entry({"state": "unavailable", "supported": False, "reason": "Not supported on this"})
    assert mac["state"] == "not_supported"
    assert model_health.gnomon_entry({"state": "disabled"})["state"] == "off"
    assert model_health.gnomon_entry({"state": "ready", "device": "cuda"})["device"] == "cuda"


def test_voice_on_the_cpu_because_the_card_was_hot_says_so():
    voice = VoiceConfig(tts_engine="kokoro_tts", tts_device="auto")
    entry = model_health.voice_entry(
        voice, _manager(running=True, engine="kokoro_tts", loaded=True, effective_device="cpu"), None
    )
    assert entry["state"] == "loaded" and entry["device"] == "cpu" and "hot, busy or switched off" in entry["reason"]
    failing = model_health.voice_entry(voice, _manager(failing=True), {"component": "voice", "message": "exit"})
    assert failing["state"] == "unavailable" and failing["last_fault"]["message"] == "exit"
    system = model_health.voice_entry(VoiceConfig(tts_engine="os_native"), _manager(), None)
    assert system["state"] == "loaded" and system["device"] == "system"


def test_recognisers_from_the_listener_stats():
    assert {entry["state"] for entry in model_health.recogniser_entries(None, [])} == {"off"}
    stats = {"running": True, "streaming_ready": True, "turn_detector_ready": False,
             "turn_detector_error": "onnxruntime missing", "final_pass": "whisper", "final_pass_notice": ""}  # fmt: skip
    entries = {entry["name"]: entry for entry in model_health.recogniser_entries(stats, ["cuda"])}
    assert entries["Streaming recogniser"]["state"] == "loaded"
    assert entries["Turn detector"]["reason"] == "onnxruntime missing"
    assert entries["Final pass (Whisper)"]["device"] == "cuda"


@pytest.mark.parametrize(
    ("platform", "tesseract", "state"),
    [
        ("windows", None, "loaded"),
        ("macos", None, "loaded"),
        ("linux", "/usr/bin/tesseract", "loaded"),
        ("linux", None, "not_supported"),
    ],  # fmt: skip
)
def test_screen_text_per_system(monkeypatch, platform, tesseract, state):
    monkeypatch.setattr(model_health.shutil, "which", lambda _name: tesseract)
    entry = model_health.ocr_entry(platform, [], None)
    assert set(entry) == KEYS and entry["state"] == state
    if state == "not_supported":
        assert "install Tesseract" in entry["reason"]
    assert model_health.ocr_entry(platform, ["gpu"], None)["device"] == "cuda"


def test_autonomy_review_status_carries_every_local_model_and_the_gpu(monkeypatch):
    monkeypatch.setattr(
        gpu_arbiter, "_shared", GpuArbiter(read=lambda: MachineReading("cpu", host_free_bytes=1), holder=lambda: None)
    )
    server = PilotServer(PilotConfig())
    status = asyncio.run(server._handle_autonomy_review_status({}, SimpleNamespace()))
    names = [entry["name"] for entry in status["local_models"]]
    assert names[:2] == ["Gnomon", "Voice"] and any(name.startswith("Screen text") for name in names)
    assert all(set(entry) == KEYS for entry in status["local_models"])
    assert status["gpu"]["machine"]["kind"] == "cpu" and status["gpu"]["disabled"] is False


async def test_the_voice_starts_on_the_cpu_when_the_arbiter_says_the_card_is_hot_and_counts_gpu_faults():
    from tests.test_local_voice_lifecycle import FakeClock, FakeWorker

    calls, faults = [], []

    def spawn(engine, device, gpu_lease=False):
        calls.append((device, gpu_lease))
        return worker

    clock = FakeClock()
    worker = FakeWorker(load_reply=CUDA_FAULT)
    manager = LocalTTSWorkerManager(spawn=spawn, clock=clock, sleep=clock.sleep)
    manager.choose_device = lambda configured: ("cpu", False)  # hot card
    manager.on_gpu_fault = lambda component, reason: faults.append(component)
    with pytest.raises(LocalTTSUnavailable):
        await manager.ensure_ready("kokoro_tts", "af_heart")
    assert calls == [("cpu", False)] and faults == []  # a fault on the CPU is not a GPU fault

    worker = FakeWorker(load_reply=CUDA_FAULT)
    await clock.advance(200)
    manager.choose_device = lambda configured: (configured, True)  # cool and free: take the lease
    with pytest.raises(LocalTTSUnavailable):
        await manager.ensure_ready("kokoro_tts", "af_heart")
    assert calls[-1] == ("auto", True) and faults == ["voice"]
    await manager.shutdown()
