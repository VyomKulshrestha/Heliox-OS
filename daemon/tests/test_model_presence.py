"""No speech model is ever downloaded by itself. Each combination of present and absent models, with a guard that
fails the test if anything opens a network connection."""

from __future__ import annotations

import asyncio
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system import model_presence, speech_models
from pilot.system.model_presence import SpeechModelMissing, wake_route


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Anything leaving this computer fails the test (the event loop's own local socket pair is allowed)."""
    original_connect = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else ""
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"a test tried to reach the network: {address!r}")
        return original_connect(self, address)

    def create_connection(address, *args, **kwargs):
        raise AssertionError(f"a test tried to reach the network: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A computer with nothing on it; tests add what they need."""
    state = {"data": set(), "hf": set(), "openai": set()}
    monkeypatch.setattr(speech_models, "models_dir", lambda data_dir=None: tmp_path / "models")

    def local_path(model_id, data_dir=None):
        return tmp_path / "models" / model_id if model_id in state["data"] else None

    monkeypatch.setattr(speech_models, "local_path", local_path)
    monkeypatch.setattr(
        model_presence, "hf_cached", lambda repo, name: f"/hf/{repo}/{name}" if repo in state["hf"] else None
    )
    monkeypatch.setattr(
        model_presence, "openai_whisper_cached", lambda name: f"/cache/{name}.pt" if name in state["openai"] else None
    )
    return state


def test_whisper_is_found_in_the_data_folder_or_an_existing_cache_and_nowhere_else(machine):
    assert model_presence.faster_whisper_source("base") is None
    machine["hf"].add("Systran/faster-whisper-base")
    assert model_presence.faster_whisper_source("base") == "base"  # loaded with local files only
    machine["data"].add("whisper-base")
    assert model_presence.faster_whisper_source("base").endswith("whisper-base")  # the app's own download first
    assert not model_presence.whisper_present("small")
    machine["openai"].add("small")
    assert model_presence.whisper_present("small")


def test_a_missing_faster_whisper_model_raises_and_is_never_fetched(machine, monkeypatch):
    from pilot.system import voice

    made = []
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=lambda *a, **k: made.append(k)))
    monkeypatch.setattr(voice, "_faster_whisper_runtime", lambda name="base": ("cpu", "int8"))
    voice._faster_whisper_model_cache.clear()
    with pytest.raises(SpeechModelMissing):
        voice._get_faster_whisper_model("small")
    assert made == []
    machine["hf"].add("Systran/faster-whisper-small")
    voice._get_faster_whisper_model("small")
    assert made[0]["local_files_only"] is True
    voice._faster_whisper_model_cache.clear()


def test_openai_whisper_loads_only_a_file_already_in_its_cache(machine, monkeypatch):
    from pilot.system import voice

    loaded = []
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda name: loaded.append(name)))
    voice._whisper_model_cache.clear()
    with pytest.raises(SpeechModelMissing):
        voice._get_whisper_model("small")
    machine["openai"].add("small")
    voice._get_whisper_model("small")
    assert loaded == ["/cache/small.pt"]  # a path, so the loader has nothing to download
    voice._whisper_model_cache.clear()


def test_captions_and_smart_turn_open_only_local_files(monkeypatch):
    from pilot.system import streaming_asr, turn_detector

    calls = []

    def hf_hub_download(repo, name, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not in the cache")  # what local_files_only gives when the file is absent

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf_hub_download))
    monkeypatch.setitem(sys.modules, "sherpa_onnx", MagicMock())
    monkeypatch.setitem(sys.modules, "onnxruntime", MagicMock())
    monkeypatch.setitem(sys.modules, "faster_whisper.feature_extractor", SimpleNamespace(FeatureExtractor=MagicMock()))
    monkeypatch.setattr(speech_models, "local_path", lambda model_id, data_dir=None: None)
    assert streaming_asr.StreamingTranscriber().load() is False
    assert turn_detector.SmartTurnDetector().load() is False
    assert calls and all(kwargs.get("local_files_only") is True for kwargs in calls)


def listener_with(present: dict[str, bool]):
    from pilot.system.voice import ContinuousVoiceListener

    statuses = []

    async def on_status(name, data):
        statuses.append((name, data))

    config = PilotConfig()
    config.voice.final_pass = "parakeet"
    item = ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=config)
    item.recogniser_presence = lambda: dict(present)
    item._loop = asyncio.get_running_loop()
    return item, statuses


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("present", "said"),
    [
        ({"whisper": False, "parakeet": False}, True),
        ({"whisper": True, "parakeet": False}, False),
        ({"whisper": False, "parakeet": True}, False),
        ({"whisper": True, "parakeet": True}, False),
    ],
)
async def test_with_no_recogniser_the_voice_bar_says_so_once_and_points_to_settings(present, said):
    item, statuses = listener_with(present)
    item._note_recogniser_missing()
    item._note_recogniser_missing()
    await asyncio.sleep(0.05)
    missing = [data for name, data in statuses if name == "recogniser_missing"]
    assert len(missing) == (1 if said else 0)
    if said:
        assert missing[0]["settings"] == "voice-models" and "Typing works" in missing[0]["message"]
        offer = missing[0]["recommended"]  # the set for the language setting, offered with its size
        assert offer["missing"] and f"{round(offer['missing_bytes'] / 1_048_576)} MB" in missing[0]["message"]


@pytest.mark.asyncio
async def test_with_only_parakeet_its_words_still_count_and_the_wake_phrase_comes_from_them(monkeypatch):
    import numpy as np

    from pilot.system import voice

    item, statuses = listener_with({"whisper": False, "parakeet": True})
    fake = MagicMock()
    fake.transcribe.return_value = "Hey Heliox, what time is it?"
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    monkeypatch.setattr(voice, "_transcribe_speech", AsyncMock(side_effect=SpeechModelMissing("no whisper")))
    text = await item._transcribe_path("utterance.wav")
    assert text == "Hey Heliox, what time is it?"
    assert voice.loose_wake_prefix(text) == "Hey Heliox"
    await asyncio.sleep(0.05)
    assert not [name for name, _ in statuses if name == "recogniser_missing"]  # Parakeet hears: nothing to say


@pytest.mark.asyncio
async def test_with_no_recogniser_an_utterance_gives_no_words_and_no_download(monkeypatch):
    from pilot.system import voice

    item, statuses = listener_with({"whisper": False, "parakeet": False})
    monkeypatch.setattr(item, "_parakeet", lambda: None)
    monkeypatch.setattr(voice, "_transcribe_speech", AsyncMock(side_effect=SpeechModelMissing("no whisper")))
    assert await item._transcribe_path("utterance.wav") == ""
    await asyncio.sleep(0.05)
    assert [name for name, _ in statuses] == ["recogniser_missing"]


def test_the_wake_route_is_said_for_each_combination():
    assert "Whisper listens for the wake phrase" in wake_route({"whisper": True, "parakeet": True})
    assert wake_route({"whisper": True, "parakeet": False}).startswith("Whisper hears the words and the wake")
    assert "missed more often" in wake_route({"whisper": False, "parakeet": True})
    assert "cannot hear the wake phrase" in wake_route({"whisper": False, "parakeet": False})
