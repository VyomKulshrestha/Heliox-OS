"""The voice a new install speaks with, chosen by the computer and the language (pilot.system.voice_default).

Made-up machines only: each carries its own reading of the card, PyTorch build, environment, caches and language. No
model loads, nothing is fetched, no real cache or Heliox folder is read or written."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import voice_default
from pilot.system.gpu_arbiter import GIB, GpuInfo, MachineReading
from pilot.system.voice_default import first_voice, machine_kind, pick, suggestion

NEEDS = int(1.4 * GIB)  # the voice's 900 MB and the arbiter's 512 MB headroom
CARD = MachineReading("nvidia", (GpuInfo(0, "NVIDIA GeForce RTX 4060 Laptop GPU", 60.0, 7 * GIB, 8 * GIB),))
NO_CARD = MachineReading("cpu")
MAC = MachineReading("apple")
EVERY_LANGUAGE = ["en", "es", "fr", "it", "pt", "hi", "ja", "zh", "de", "nl", "sw"]


# ── The table ──


def test_the_table_is_data_and_reads_as_decided():
    rows = [(rule.machine, rule.languages, rule.engine) for rule in voice_default.RULES]
    assert rows == [
        ("nvidia_card", ("*",), "kokoro_tts"),
        ("apple_silicon", ("en",), "pocket_tts"),
        ("apple_silicon", ("*",), "kokoro_tts"),
        ("processor", ("en",), "pocket_tts"),
        ("processor", ("es", "fr", "it", "pt", "hi", "ja", "zh"), "kokoro_tts"),
        ("processor", ("*",), "kokoro_tts"),
    ]
    assert all(rule.why for rule in voice_default.RULES)


@pytest.mark.parametrize("language", EVERY_LANGUAGE)
def test_each_language_on_each_kind_of_computer(language):
    assert pick("nvidia_card", language).engine == "kokoro_tts"
    assert pick("apple_silicon", language).engine == ("pocket_tts" if language == "en" else "kokoro_tts")
    assert pick("processor", language).engine == ("pocket_tts" if language == "en" else "kokoro_tts")


def test_a_regional_code_reads_as_its_language():
    assert pick("processor", "en-GB").engine == "pocket_tts" and pick("processor", "EN_us").engine == "pocket_tts"
    assert pick("processor", "").engine == "pocket_tts"  # nothing known: English, as voice_words takes it


def test_changing_one_row_is_the_whole_change(tmp_path):
    data = json.loads(voice_default.MANIFEST.read_text(encoding="utf-8"))
    data["voice_rules"]["rows"][4]["languages"] = ["fr", "it", "pt", "hi", "ja", "zh"]
    data["voice_rules"]["rows"].insert(4, {"machine": "processor", "languages": ["es"], "engine": "pocket_tts"})
    data["voice_rules"]["rows"][2]["engine"] = "pocket_tts"  # say Pocket passes French on a Mac too
    manifest = tmp_path / "speech_models.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    rules = voice_default.load_rules(manifest)
    assert (
        pick("processor", "es", rules).engine == "pocket_tts" and pick("processor", "fr", rules).engine == "kokoro_tts"
    )
    assert pick("apple_silicon", "fr", rules).engine == "pocket_tts"
    data["voice_rules"]["rows"][0]["engine"] = "piper"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        voice_default.load_rules(manifest)


def test_the_voices_say_what_they_are_their_size_licence_and_source():
    pocket, kokoro = voice_default.VOICES["pocket_tts"], voice_default.VOICES["kokoro_tts"]
    assert pocket["licence"] == "CC-BY-4.0" and "kyutai/pocket-tts-without-voice-cloning" in pocket["source"]
    assert "kyutai/tts-voices" not in pocket["source"] and "CC BY 4.0" in pocket["attribution"]
    assert pocket["voice_terms"]["alba"]["source"] == "kyutai/tts-voices alba-mackenna/casual.wav"
    assert 219_000_000 <= pocket["download_bytes"] <= 230_000_000 and pocket["default_voice"] == "alba"
    assert kokoro["licence"] == "Apache-2.0" and "hexgrad/Kokoro-82M" in kokoro["source"]
    assert kokoro["default_voice"] == "af_heart" and pocket["languages"] == ["en"]


# ── Which kind of computer ──


@pytest.mark.parametrize(
    ("reading", "torch", "extra", "kind"),
    [
        (CARD, "cuda", {}, "nvidia_card"),  # Windows with Voice on the graphics card, or Linux with a card
        (CARD, "cpu", {}, "processor"),  # Windows with a card but PyPI's PyTorch (a processor build)
        (CARD, "", {}, "processor"),  # Voice not installed yet
        (NO_CARD, "cuda", {}, "processor"),  # Linux without a card: PyPI's PyTorch is the CUDA build anyway
        (NO_CARD, "cpu", {}, "processor"),
        (MAC, "mps", {}, "apple_silicon"),
        (MAC, "mps", {"tts_device": "cpu"}, "apple_silicon"),
        (CARD, "cuda", {"hidden": True}, "processor"),  # CUDA_VISIBLE_DEVICES hides every card
        (CARD, "cuda", {"tts_device": "cpu"}, "processor"),  # the voice is set to the processor
        (MachineReading("nvidia", CARD.gpus, error="timeout"), "cuda", {}, "processor"),
        (MachineReading("nvidia", (GpuInfo(0, "GT 710", None, None, 1 * GIB),)), "cuda", {}, "processor"),
        (MachineReading("nvidia", (GpuInfo(0, "unknown", None, None, None),)), "cuda", {}, "nvidia_card"),
        (MachineReading("nvidia", (GpuInfo(0, "hot", 95.0, 0, 8 * GIB),)), "cuda", {}, "nvidia_card"),  # heat passes
    ],
)
def test_the_kind_of_computer_from_its_card_and_pytorch(reading, torch, extra, kind):
    assert machine_kind(reading, torch_build=torch, needs_bytes=NEEDS, **extra) == kind


def test_the_voice_reads_the_first_visible_card():
    small_then_big = MachineReading(
        "nvidia", (GpuInfo(0, "small", None, None, GIB), GpuInfo(1, "big", None, None, 8 * GIB))
    )
    assert machine_kind(small_then_big, torch_build="cuda", needs_bytes=NEEDS) == "processor"
    assert machine_kind(small_then_big, torch_build="cuda", needs_bytes=NEEDS, visible=(1, 0)) == "nvidia_card"
    assert machine_kind(small_then_big, torch_build="cuda", needs_bytes=NEEDS, visible=()) == "processor"


# ── What is on disk ──


def test_the_hugging_face_cache_is_found_as_huggingface_hub_finds_it(tmp_path):
    home = tmp_path / "home"
    assert voice_default.hf_cache_dir({}, home) == home / ".cache" / "huggingface" / "hub"
    assert (
        voice_default.hf_cache_dir({"XDG_CACHE_HOME": str(tmp_path / "x")}, home)
        == tmp_path / "x" / "huggingface" / "hub"
    )
    assert voice_default.hf_cache_dir({"HF_HOME": str(tmp_path / "h")}, home) == tmp_path / "h" / "hub"
    both = {"HF_HOME": str(tmp_path / "h"), "HF_HUB_CACHE": str(tmp_path / "c")}
    assert voice_default.hf_cache_dir(both, home) == tmp_path / "c"


def _put(cache: Path, relative: str) -> None:
    path = cache / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_a_model_counts_as_on_disk_only_once_its_weights_are_there(tmp_path):
    cache = tmp_path / "hub"
    assert voice_default.voice_models_on_disk(cache) == frozenset()
    _put(cache, "models--hexgrad--Kokoro-82M/snapshots/abc/config.json")  # a fetch that stopped before the weights
    _put(cache, "models--kyutai--pocket-tts-without-voice-cloning/snapshots/d/languages/english/tokenizer.json")
    assert voice_default.voice_models_on_disk(cache) == frozenset()
    _put(cache, "models--hexgrad--Kokoro-82M/snapshots/abc/kokoro-v1_0.pth")
    assert voice_default.voice_models_on_disk(cache) == {"kokoro_tts"}
    _put(cache, "models--kyutai--pocket-tts-without-voice-cloning/snapshots/d/languages/english/model.safetensors")
    assert voice_default.voice_models_on_disk(cache) == {"kokoro_tts", "pocket_tts"}
    assert not voice_default.pocket_voice_on_disk("alba", cache)
    _put(
        cache,
        "models--kyutai--pocket-tts-without-voice-cloning/snapshots/d/languages/english/embeddings/alba.safetensors",
    )
    assert voice_default.pocket_voice_on_disk("alba", cache) and not voice_default.pocket_voice_on_disk("", cache)


def test_a_voice_that_has_loaded_here_is_known_from_the_voice_store(tmp_path):
    store = tmp_path / "voice_heads"
    assert voice_default.voice_ever_loaded(store) is False
    store.mkdir()
    (store / "loads.json").write_text("[]", encoding="utf-8")
    (store / "index.json").write_text("not json", encoding="utf-8")
    assert voice_default.voice_ever_loaded(store) is False
    (store / "loads.json").write_text(json.dumps([["kokoro_tts", "cpu", 10.7]]), encoding="utf-8")
    assert voice_default.voice_ever_loaded(store) is True


# ── The first voice ──


def fresh(**given):
    settings = {"engine": "kokoro_tts", "voice": "af_heart", "chosen": False, "machine": "processor", "language": "en",
                "models_on_disk": frozenset(), "ever_loaded": False}  # fmt: skip
    settings.update(given)
    return first_voice(**settings)


def test_a_new_english_install_without_a_card_starts_with_pocket():
    choice = fresh()
    assert (choice.engine, choice.voice, choice.machine) == ("pocket_tts", "alba", "processor")
    assert "0.6 s sooner" in choice.why


@pytest.mark.parametrize("language", EVERY_LANGUAGE)
def test_every_other_new_install_keeps_kokoro(language):
    assert fresh(machine="nvidia_card", language=language) is None
    if language != "en":
        assert fresh(language=language) is None
        assert fresh(machine="apple_silicon", language=language) is None


def test_a_new_english_install_on_apple_silicon_starts_with_pocket():
    choice = fresh(machine="apple_silicon")
    assert (choice.engine, choice.voice, choice.machine) == ("pocket_tts", "alba", "apple_silicon")
    assert "Apple M1 runner" in choice.why and "Kokoro has not been measured on a Mac" in choice.why


def test_an_install_with_a_voice_of_its_own_is_never_switched():
    assert fresh(chosen=True) is None  # the person chose it
    assert fresh(engine="os_native", voice="") is None  # the system voice
    assert fresh(models_on_disk=frozenset({"kokoro_tts"})) is None  # Kokoro is already here: it has spoken
    assert fresh(models_on_disk=frozenset({"pocket_tts"})) is None
    assert fresh(ever_loaded=True) is None  # a voice has loaded here, even if its model was deleted since


def test_an_install_still_without_a_voice_follows_new_hardware():
    """The table was applied on an earlier start, nothing was fetched yet (no network), then a card arrived."""
    choice = fresh(engine="pocket_tts", voice="alba", machine="nvidia_card")
    assert (choice.engine, choice.voice) == ("kokoro_tts", "af_heart")
    assert fresh(engine="pocket_tts", voice="alba") is None  # the same computer: nothing to change


def test_a_face_without_an_approved_pocket_voice_keeps_kokoro_and_its_own_voice():
    man = {"kokoro_tts": "am_michael", "pocket_tts": ""}  # a kind Pocket has no approved voice for
    woman = {"kokoro_tts": "af_heart", "pocket_tts": "alba"}
    assert fresh(face_voices=man) is None  # Kokoro stays; the face then brings am_michael
    back = fresh(engine="pocket_tts", voice="alba", face_voices=man)
    assert (back.engine, back.voice) == ("kokoro_tts", "am_michael")
    assert (fresh(face_voices=woman).engine, fresh(face_voices=woman).voice) == ("pocket_tts", "alba")


# ── The suggestion for an install whose voice already speaks ──


def suggest(**given):
    settings = {"engine": "kokoro_tts", "machine": "processor", "language": "en", "small_machine": False,
                "pocket_installed": True, "pocket_on_disk": False}  # fmt: skip
    settings.update(given)
    return suggestion(**settings)


def test_kokoro_on_the_processor_in_english_is_told_once_what_pocket_would_change():
    said = suggest()
    assert said["engine"] == "pocket_tts" and said["voice"] == "alba" and said["download_bytes"] == 225_000_000
    assert said["say"].startswith("On this computer Pocket TTS would answer about 0.6 s sooner than Kokoro")
    assert "about 225 MB" in said["say"] and "Your voice stays as it is unless you switch." in said["say"]
    here = suggest(pocket_on_disk=True)
    assert here["download_bytes"] == 0 and "already on this computer" in here["say"]


@pytest.mark.parametrize(
    "given",
    [
        {"engine": "pocket_tts"},
        {"engine": "os_native"},
        {"machine": "nvidia_card"},
        {"machine": "apple_silicon"},
        {"language": "es"},
        {"language": "de"},
        {"small_machine": True},  # 8 GB: the system voice speaks anyway
        {"pocket_installed": False},
    ],
)
def test_no_suggestion_where_it_would_not_help(given):
    assert suggest(**given) is None


# ── The daemon applies it at start ──


def _server(monkeypatch, *, language="en", chosen=False, engine="kokoro_tts", voice="af_heart", pack="off"):
    config = PilotConfig()
    config.voice.language = language
    config.voice.tts_engine, config.voice.tts_voice, config.voice.tts_voice_chosen = engine, voice, chosen
    config.face.pack = pack
    config.save = MagicMock()
    return PilotServer(config), config


def _probes(kind="processor", on_disk=frozenset(), loaded=False):
    asked: list[str] = []

    def machine(device):
        asked.append(device)
        return kind

    return asked, {"machine": machine, "models_on_disk": lambda: on_disk, "ever_loaded": lambda: loaded}


def test_the_daemon_writes_the_first_voice_once_and_saves_it(monkeypatch):
    server, config = _server(monkeypatch)
    asked, probes = _probes()
    written = server._choose_first_voice(**probes)
    assert written == {"engine": "pocket_tts", "voice": "alba", "why": written["why"], "machine": "processor"}
    assert (config.voice.tts_engine, config.voice.tts_voice) == ("pocket_tts", "alba")
    assert config.voice.tts_voice_chosen is False and config.save.call_count == 1 and asked == ["auto"]
    # The next start, with Pocket fetched: nothing more changes.
    _asked, probes = _probes(on_disk=frozenset({"pocket_tts"}))
    assert server._choose_first_voice(**probes) is None and config.save.call_count == 1


@pytest.mark.parametrize(("kind", "language"), [("nvidia_card", "en"), ("apple_silicon", "fr"), ("processor", "es"),
                                                ("processor", "de"), ("processor", "ja")])  # fmt: skip
def test_the_daemon_leaves_kokoro_where_the_table_says_so(monkeypatch, kind, language):
    server, config = _server(monkeypatch, language=language)
    _asked, probes = _probes(kind)
    assert server._choose_first_voice(**probes) is None
    assert config.voice.tts_engine == "kokoro_tts" and config.save.call_count == 0


def test_the_daemon_never_reads_the_card_for_an_install_with_a_voice_of_its_own(monkeypatch):
    for given in ({"chosen": True}, {"engine": "os_native"}):
        server, config = _server(monkeypatch, **given)
        asked, probes = _probes()
        assert server._choose_first_voice(**probes) is None and asked == [] and config.save.call_count == 0
    server, config = _server(monkeypatch)
    asked, probes = _probes(loaded=True)
    assert server._choose_first_voice(**probes) is None and asked == []


def test_a_failed_save_changes_nothing(monkeypatch):
    server, config = _server(monkeypatch)
    config.save.side_effect = OSError("disk full")
    _asked, probes = _probes()
    assert server._choose_first_voice(**probes) is None
    assert (config.voice.tts_engine, config.voice.tts_voice) == ("kokoro_tts", "af_heart")


def test_the_man_s_face_starts_with_michael_at_first_run(monkeypatch):
    server, config = _server(monkeypatch, pack="portrait-man")
    _asked, probes = _probes()
    assert server._choose_first_voice(**probes)["voice"] == "michael" and config.voice.tts_engine == "pocket_tts"
    server, config = _server(monkeypatch, pack="portrait-woman")
    assert server._choose_first_voice(**probes)["voice"] == "alba"


def test_start_chooses_the_voice_before_the_face_brings_its_own_and_before_the_voice_is_set_up():
    source = inspect.getsource(PilotServer.start)
    chosen = source.index("self._choose_first_voice")
    assert 0 < chosen < source.index("_face_brings_its_voice(configure=False)") < source.index("self._set_up_voice()")
    # The Conversation Bench sets the voice up with _set_up_voice alone: the table never changes what it measures.
    assert "_choose_first_voice" not in inspect.getsource(PilotServer._set_up_voice)
