from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import pilot.system.voice as voice


@pytest.fixture(autouse=True)
def _clear_model_cache(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # made-up cards: unhide what the suite hides
    voice._faster_whisper_model_cache.clear()
    # These tests are about devices and decoding: the model counts as on this computer (presence has its own tests).
    monkeypatch.setattr("pilot.system.model_presence.faster_whisper_source", lambda name: name)
    yield
    voice._faster_whisper_model_cache.clear()


def test_faster_whisper_model_is_cached_by_runtime():
    model_type = MagicMock(return_value=object())
    module = SimpleNamespace(WhisperModel=model_type)

    with (
        patch.dict("sys.modules", {"faster_whisper": module}),
        patch.object(voice, "_faster_whisper_runtime", return_value=("cpu", "int8")),
    ):
        first = voice._get_faster_whisper_model("base")
        second = voice._get_faster_whisper_model("base")

    assert first is second
    model_type.assert_called_once()
    args, kwargs = model_type.call_args
    assert args == ("base",) and kwargs["device"] == "cpu" and kwargs["compute_type"] == "int8"
    assert 1 <= kwargs["cpu_threads"] <= 8  # measured: 8 threads halve the pass on a 6 s sentence


@pytest.mark.asyncio
async def test_faster_whisper_uses_command_aware_deterministic_decoding():
    model = MagicMock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(text=" open "), SimpleNamespace(text=" GitHub ")]),
        SimpleNamespace(language="en"),
    )

    with patch.object(voice, "_get_faster_whisper_model", return_value=model):
        result = await voice._transcribe_faster_whisper("voice.wav", "auto", "base")

    assert result == {"text": "open GitHub", "language": "en"}
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["beam_size"] == 5
    assert kwargs["temperature"] == 0
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False
    assert kwargs["initial_prompt"] == voice._VOICE_TRANSCRIPTION_PROMPT
    assert kwargs["without_timestamps"] is True and "hotwords" not in kwargs
    assert "language" not in kwargs


@pytest.mark.asyncio
async def test_auto_engine_falls_back_to_openai_whisper():
    expected = {"text": "show system information", "language": "en"}
    with (
        patch.object(voice, "_transcribe_faster_whisper", side_effect=ImportError),
        patch.object(voice, "_transcribe_whisper", return_value=expected) as fallback,
    ):
        result = await voice._transcribe_speech("voice.wav", "auto", "base", "auto")

    assert result == expected
    fallback.assert_awaited_once_with("voice.wav", "auto", "base")


@pytest.mark.asyncio
async def test_explicit_faster_whisper_does_not_hide_backend_failure():
    with (
        patch.object(voice, "_transcribe_faster_whisper", side_effect=ImportError),
        pytest.raises(ImportError),
    ):
        await voice._transcribe_speech("voice.wav", "auto", "base", "faster_whisper")


@pytest.mark.asyncio
async def test_unknown_transcription_engine_is_rejected():
    with pytest.raises(ValueError, match="Unsupported transcription engine"):
        await voice._transcribe_speech("voice.wav", "auto", "base", "remote_magic")


# ── where Whisper runs: the GPU arbiter decides, and a CUDA error is a fault ──


@pytest.fixture
def gpu_world(monkeypatch):
    """One NVIDIA card as ctranslate2 and the arbiter see it; the test sets its temperature and holder."""
    from pilot.system import gpu_arbiter

    card = {"temperature": 60.0, "holder": None, "free": 6 * gpu_arbiter.GIB}
    shared = gpu_arbiter.GpuArbiter(
        read=lambda: gpu_arbiter.MachineReading(
            "nvidia",
            gpus=(gpu_arbiter.GpuInfo(0, "RTX 4060", card["temperature"], card["free"], 8 * gpu_arbiter.GIB),),
        ),
        holder=lambda: card["holder"],
    )
    monkeypatch.setattr(gpu_arbiter, "_shared", shared)
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(voice, "_faster_whisper_cuda_unusable", False)
    # Its CUDA 12 libraries are there (the real machine's are not read: that is test_gpu_paths).
    monkeypatch.setattr("pilot.system.gpu_paths.whisper_gpu_missing", lambda: [])
    return card, shared


def test_whisper_uses_the_card_only_when_the_arbiter_says_it_is_healthy(gpu_world):
    card, _shared = gpu_world
    assert voice._faster_whisper_runtime("base") == ("cuda", "float16")
    card["temperature"] = 88.0
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")
    assert "°C" in voice._faster_whisper_device_reason
    card["temperature"], card["holder"] = 60.0, {"component": "gnomon", "pid": 1}
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")  # Gnomon holds the card
    assert "Gnomon" in voice._faster_whisper_device_reason
    card["holder"], card["free"] = None, 200 * 1024**2
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")  # short of graphics memory


def test_without_a_cuda_card_whisper_is_on_the_cpu(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setattr(voice, "_faster_whisper_cuda_unusable", False)
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", None)  # not installed
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")


def test_a_cuda_error_while_loading_counts_as_a_fault_and_moves_to_the_cpu(gpu_world):
    _card, shared = gpu_world
    made = []

    def whisper_model(name, device, compute_type, **_options):
        made.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA failed with error unspecified launch failure")
        return SimpleNamespace(device=device)

    with patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=whisper_model)}):
        model = voice._get_faster_whisper_model("base")
    assert model.device == "cpu" and made == ["cuda", "cpu"]
    assert shared.last_fault("whisper") is not None
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")  # for the rest of the session


@pytest.mark.asyncio
async def test_a_cuda_error_while_transcribing_frees_the_card_and_retries_on_the_cpu(gpu_world, monkeypatch):
    _card, shared = gpu_world
    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError("CUDA failed with error out of memory")
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (iter([SimpleNamespace(text=" open notes ")]), SimpleNamespace(language="en"))

    def whisper_model(name, device, compute_type, **_options):
        return gpu_model if device == "cuda" else cpu_model

    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: None)
    with patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=whisper_model)}):
        result = await voice._transcribe_faster_whisper("utterance.wav", "en", "base")
    assert result["text"] == "open notes"
    assert not any(key[1] == "cuda" for key in voice._faster_whisper_model_cache)
    assert len(shared.faults()) == 1


@pytest.mark.asyncio
async def test_a_missing_cuda_library_while_transcribing_is_not_installed_and_never_a_fault(gpu_world, monkeypatch):
    """12 November, the owner's laptop: a healthy card, and no cublas64_12.dll. The processor for Whisper, said in
    the health status as "not installed", and nothing counted against the card."""
    from pilot.system import gpu_paths

    _card, shared = gpu_world
    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (iter([SimpleNamespace(text=" open notes ")]), SimpleNamespace(language="en"))

    def whisper_model(name, device, compute_type, **_options):
        return gpu_model if device == "cuda" else cpu_model

    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: None)
    with patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=whisper_model)}):
        result = await voice._transcribe_faster_whisper("utterance.wav", "en", "base")
    assert result["text"] == "open notes"
    assert shared.faults() == [] and gpu_paths.not_installed("whisper") == "cublas64_12.dll"
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")
    assert "not installed" in voice._faster_whisper_device_reason and "cublas64_12.dll" in (
        voice._faster_whisper_device_reason
    )


def test_a_card_without_its_cuda_libraries_is_not_tried_at_all(gpu_world, monkeypatch):
    from pilot.system import gpu_paths

    _card, shared = gpu_world
    monkeypatch.setattr(gpu_paths, "whisper_gpu_missing", lambda: ["cublas64_12.dll", "cudnn64_9.dll"])
    assert voice._faster_whisper_runtime("base") == ("cpu", "int8")
    assert shared.faults() == []
    assert voice._faster_whisper_device_reason.startswith("On the processor, because its GPU libraries are not")


def test_a_healthy_card_with_its_libraries_says_so(gpu_world):
    assert voice._faster_whisper_runtime("base") == ("cuda", "float16")
    assert voice._faster_whisper_device_reason == "On the graphics card."
