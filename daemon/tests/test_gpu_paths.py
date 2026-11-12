"""A GPU path that is not installed is not a fault of the card. For Whisper, the voice worker and Parakeet: a missing
library, runtime build or provider is said once in the health status with what is missing, and never counted by the
GPU arbiter; a real device error stays a fault. The health status says per model where it runs and why. Fakes only:
no card, no CUDA, no model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.system import gpu_paths, model_health
from pilot.system.local_tts_worker import LocalTTSWorkerManager
from pilot.system.parakeet_asr import ParakeetTranscriber, choose_provider


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        ("Library cublas64_12.dll is not found or cannot be loaded", "cublas64_12.dll"),
        ("Could not load library cudnn_ops64_9.dll. Error code 126", "a GPU library"),
        ("libcudnn.so.9: cannot open shared object file: No such file or directory", "libcudnn.so.9"),
        ("AssertionError: Torch not compiled with CUDA enabled", "a CUDA build of PyTorch"),
        ("CUDAExecutionProvider is not in available provider names", "the GPU build of the ONNX runtime"),
        ("CUDA driver version is insufficient for CUDA runtime version", "a newer NVIDIA driver"),
        ("ImportError: DLL load failed while importing _C", "a GPU library"),
    ],
)
def test_errors_that_mean_not_installed(text, missing):
    assert gpu_paths.missing_library(text) == missing


@pytest.mark.parametrize(
    "text",
    [
        "RuntimeError: CUDA error: unknown error",
        "RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
        "CUDA failed with error out of memory",
        "CUDA failed with error unspecified launch failure",
        "the worker exited",
    ],
)
def test_device_errors_stay_faults(text):
    assert gpu_paths.missing_library(text) == ""


def fake_torch(monkeypatch, version, requires=()):
    dist = SimpleNamespace(version=version, requires=list(requires))
    monkeypatch.setattr(gpu_paths.importlib.metadata, "distribution", lambda name: dist)


@pytest.mark.parametrize(
    ("version", "requires", "platform", "build"),
    [
        ("2.6.0", [], "win32", "cpu"),  # PyPI's Windows wheel
        ("2.6.0+cu126", [], "win32", "cuda"),  # the CUDA index's
        ("2.6.0", ["nvidia-cublas-cu12==12.4.5.8; platform_system == 'Linux'"], "linux", "cuda"),  # PyPI Linux
        ("2.6.0+cpu", [], "linux", "cpu"),
        ("2.6.0", [], "darwin", "mps"),
    ],
)
def test_the_torch_build_is_read_from_its_metadata_without_importing_it(
    monkeypatch, version, requires, platform, build
):
    fake_torch(monkeypatch, version, requires)
    assert gpu_paths.torch_build(platform) == build


def test_no_torch_is_no_build(monkeypatch):
    def missing(name):
        raise gpu_paths.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(gpu_paths.importlib.metadata, "distribution", missing)
    assert gpu_paths.torch_build("win32") == ""


@pytest.mark.parametrize(("platform", "expected"), [("win32", ["cudnn64_9.dll"]), ("linux", ["libcudnn.so.9"])])
def test_whisper_s_libraries_are_checked_by_name(monkeypatch, platform, expected):
    monkeypatch.setattr(gpu_paths, "prepare_gpu_libraries", lambda: [])
    assert gpu_paths.whisper_gpu_missing(platform, loadable=lambda name: "cudnn" not in name) == expected
    assert gpu_paths.whisper_gpu_missing("darwin", loadable=lambda name: False) == []


def test_only_the_nvidia_folders_whisper_needs_are_found(monkeypatch, tmp_path):
    """Only this test's own site-packages is searched (the Linux runners' real one holds PyTorch's NVIDIA wheels),
    and of the parts there only Whisper's: never nccl, nvshmem or cusparselt."""
    leaf = "bin" if sys.platform == "win32" else "lib"
    for part in ("cublas", "cudnn", "nccl", "nvshmem", "cusparselt"):
        (tmp_path / "nvidia" / part / leaf).mkdir(parents=True)
    (tmp_path / "nvidia" / "cuda_runtime").mkdir(parents=True)  # nothing to load there
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])
    monkeypatch.setattr(gpu_paths.importlib.util, "find_spec", lambda name: spec if name == "nvidia" else None)
    found = gpu_paths.nvidia_library_dirs()
    assert [f.parent.name for f in found] == ["cublas", "cudnn"]


def test_not_installed_is_said_once_and_forgotten_on_reset(caplog):
    caplog.set_level("INFO", logger="pilot.system.gpu_paths")
    assert gpu_paths.note_not_installed("whisper", "cublas64_12.dll")
    assert not gpu_paths.note_not_installed("whisper", "cublas64_12.dll")
    assert len([r for r in caplog.records if "not installed" in r.getMessage()]) == 1
    gpu_paths.reset()
    assert gpu_paths.not_installed("whisper") == ""


# ── The voice worker ──


class Worker:
    """A worker process whose answers the test sets: a load and a sentence."""

    def __init__(self, device: str, error: str = "") -> None:
        self.device, self.error = device, error if device != "cpu" else ""
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
        if self.error and request.get("op") != "load":
            return (json.dumps({"status": "error", "error": self.error}) + "\n").encode()
        if request.get("op") != "load":
            Path(request["output_file"]).write_bytes(b"RIFF-test-audio")
        return b'{"status":"ok"}\n'

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def voice_manager(error: str):
    started, faults, said = [], [], []

    def spawn(engine, device, gpu_lease=False):
        started.append(device)
        return Worker(device, error)

    manager = LocalTTSWorkerManager(spawn=spawn)
    manager.choose_device = lambda configured: (configured, True)
    manager.on_gpu_fault = lambda component, reason: faults.append(component)

    async def on_status(status, data):
        said.append(status)

    manager.on_status = on_status
    return manager, started, faults, said


@pytest.fixture
def cuda_torch(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_paths, "torch_build", lambda platform=None: "cuda")


async def speak(manager):
    async def play(path):
        return None

    await manager.speak("kokoro_tts", "Hello there.", "af_heart", None, play)


@pytest.mark.asyncio
async def test_a_voice_worker_missing_a_library_moves_to_the_processor_without_a_fault(cuda_torch):
    manager, started, faults, said = voice_manager("OSError: libcudnn.so.9: cannot open shared object file")
    await speak(manager)
    assert started == ["auto", "cpu"] and faults == [] and "voice_on_cpu" not in said
    assert gpu_paths.not_installed("voice") == "libcudnn.so.9"
    assert "not installed" in manager.device_reason
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_voice_worker_with_a_device_error_is_still_a_fault(cuda_torch):
    manager, started, faults, said = voice_manager("RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
    await speak(manager)
    assert started == ["auto", "cpu"] and faults == ["voice"] and "voice_on_cpu" in said
    assert gpu_paths.not_installed("voice") == ""
    assert manager.device_reason == "On the processor for this session: the graphics card failed for the voice."
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_cpu_build_of_torch_never_takes_the_card(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_paths, "torch_build", lambda platform=None: "cpu")
    manager, started, faults, _said = voice_manager("")
    await speak(manager)
    assert started == ["cpu"] and faults == []
    assert manager.device_reason.startswith("On the processor, because its GPU libraries are not installed")
    assert "a CUDA build of PyTorch" in manager.device_reason
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_voice_on_the_card_says_so(cuda_torch):
    manager, started, _faults, _said = voice_manager("")
    await speak(manager)
    assert started == ["auto"] and manager.device_reason == "On the graphics card."
    entry = model_health.voice_entry(SimpleNamespace(tts_engine="kokoro_tts", tts_device="auto"), manager, None)
    assert entry["reason"].endswith("On the graphics card.")
    await manager.shutdown()


# ── Parakeet's provider ──


def decision(kind="nvidia", use_gpu=True, reason=""):
    return lambda: SimpleNamespace(kind=kind, use_gpu=use_gpu, reason=reason)


def test_parakeet_without_a_card_or_without_its_cuda_build_stays_on_the_processor(monkeypatch):
    assert choose_provider(decision(kind="cpu")) == ("cpu", "On the processor: no NVIDIA graphics card.")
    monkeypatch.setattr(gpu_paths, "sherpa_onnx_cuda", lambda: False)
    provider, reason = choose_provider(decision())
    assert provider == "cpu" and "the CUDA build of sherpa-onnx" in reason and "not installed" in reason
    monkeypatch.setattr(gpu_paths, "sherpa_onnx_cuda", lambda: True)
    assert choose_provider(decision(use_gpu=False, reason="the card is at 88 °C."))[0] == "cpu"
    assert choose_provider(decision()) == ("cuda", "On the graphics card.")


class Sherpa:
    """sherpa_onnx as Parakeet opens it: its CUDA provider fails with ``error``; the CPU one works."""

    def __init__(self, error: str) -> None:
        self.error, self.asked = error, []
        self.OfflineRecognizer = SimpleNamespace(from_transducer=self.open)

    def open(self, **options):
        self.asked.append(options["provider"])
        if options["provider"] != "cpu":
            raise RuntimeError(self.error)
        return "recognizer"


def test_parakeet_s_provider_missing_is_not_installed_and_a_device_error_is_a_fault(tmp_path):
    faults = []
    missing = ParakeetTranscriber(tmp_path, provider="cuda", reason="On the graphics card.")
    missing.on_gpu_fault = lambda component, message: faults.append(component)
    sherpa = Sherpa("Could not find onnxruntime_providers_cuda.dll: is not found")
    assert missing._open(sherpa) == "recognizer" and sherpa.asked == ["cuda", "cpu"]
    assert missing.provider == "cpu" and faults == [] and "not installed" in missing.device_reason
    assert gpu_paths.not_installed("parakeet") == "onnxruntime_providers_cuda.dll"

    broken = ParakeetTranscriber(tmp_path, provider="cuda")
    broken.on_gpu_fault = lambda component, message: faults.append(component)
    assert broken._open(Sherpa("CUDA failure 999: unknown error")) == "recognizer"
    assert faults == ["parakeet"] and broken.device_reason.startswith("On the processor: the graphics card failed")


def test_the_health_status_says_where_each_recogniser_runs_and_why():
    stats = {
        "running": True,
        "streaming_ready": True,
        "turn_detector_ready": True,
        "final_pass": "parakeet",
        "parakeet_device": "cpu",
        "parakeet_device_reason": gpu_paths.on_processor_because_missing("the CUDA build of sherpa-onnx"),
        "whisper_device_reason": gpu_paths.on_processor_because_missing("cublas64_12.dll"),
    }
    entries = {e["name"]: e for e in model_health.recogniser_entries(stats, ["cpu"])}
    assert entries["Final pass (Parakeet)"]["device"] == "cpu"
    assert "sherpa-onnx" in entries["Final pass (Parakeet)"]["reason"]
    whisper = entries["Wake check (Whisper)"]
    assert whisper["device"] == "cpu" and whisper["reason"].startswith("On the processor, because its GPU libraries")


def test_the_graphics_card_entry_says_why_the_speech_models_are_where_they_are():
    none = model_health.gpu_entry(SimpleNamespace(kind="cpu"))
    assert none["state"] == "not_supported" and none["device"] == "cpu"
    assert none["reason"] == "No NVIDIA graphics card: speech recognition and the voice run on the processor."
    card = SimpleNamespace(kind="nvidia")
    assert model_health.gpu_entry(card)["reason"] == "Available to the speech models."
    gpu_paths.note_not_installed("whisper", "cublas64_12.dll")
    missing = model_health.gpu_entry(card)
    assert missing["state"] == "unavailable" and "Whisper: cublas64_12.dll" in missing["reason"]
    assert gpu_paths.INSTALL_HINT in missing["reason"]
    off = model_health.gpu_entry(card, "Stopped using the graphics card for this session after 3 faults.")
    assert off["state"] == "off"
    assert set(none) == set(off)


def test_the_first_run_check_reads_the_graphics_card_entry():
    import importlib.util

    path = Path(__file__).parent / "platform_real" / "first_run_rpc_check.py"
    spec = importlib.util.spec_from_file_location("first_run_rpc_check", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    runner = {"local_models": [model_health.gpu_entry(SimpleNamespace(kind="cpu"))]}
    assert check.gpu_check(runner).startswith("CHECK first-run/gpu-status: PASS - No NVIDIA graphics card")
    assert "FAIL" in check.gpu_check({"local_models": []})
