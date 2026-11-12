"""A new person's wake phrase: the teach step keeps the spellings their recogniser writes for "Hey Heliox", refuses
ordinary words, is bounded and removable; a taught spelling wakes Heliox, also with punctuation; and a new person's
Whisper is the one on their computer, not the settings default. No audio, no models, no network."""

from __future__ import annotations

import asyncio

import pytest

from pilot.config import PilotConfig
from pilot.system import model_presence
from pilot.system.voice import ContinuousVoiceListener
from pilot.system.voice_calibration import (
    MAX_TAUGHT,
    VoiceCalibrationStore,
    WakeWordCalibrator,
    refusal,
    teach,
)


def calibrator(tmp_path) -> WakeWordCalibrator:
    return WakeWordCalibrator(["hey heliox", "heliox"], VoiceCalibrationStore(data_dir=tmp_path / "calibration"))


def test_a_spelling_written_twice_is_kept_and_ordinary_words_are_refused():
    keep, refused = teach(["Hey Helios!", "Hey Helios.", "Hey Alex.", "Hey Alex", "Hey Heliox."])
    assert keep == ["hey helios"]
    assert "hey alex" in refused and "ordinary words" in refused["hey alex"]


def test_a_voice_the_recogniser_writes_differently_every_time_teaches_nothing():
    # The owner's desk microphone through Parakeet: a different spelling on every try.
    keep, refused = teach(["Hey Lux, what can I say?", "Hey Hideof", "Dear Yorks", "Hey, Yox."])
    assert keep == [] and refused  # each refused: heard once, too short, or nothing like the name


@pytest.mark.parametrize(
    "spelling",
    ["hey helix", "hello", "here it is", "hey lux", "hey alexa", "hello there", "type powershell", "dear yorks"],
)
def test_spellings_that_would_wake_heliox_in_ordinary_talk_are_refused(spelling):
    assert refusal(spelling)


@pytest.mark.parametrize("spelling", ["hey helios", "hey hilux", "hey elio", "hey heliocs"])
def test_spellings_that_are_not_ordinary_words_are_accepted(spelling):
    assert refusal(spelling) == ""


def test_taught_spellings_are_bounded_kept_on_disk_and_removable(tmp_path):
    item = calibrator(tmp_path)
    spellings = ["hey helios", "hey heliosh", "hey helioz", "hey helyox", "hey hiliox", "hey heelox", "hey heliks"]
    for spelling in spellings:
        item.add_taught([spelling])
    kept = [v.text for v in item.get_variants() if v.taught]
    assert len(kept) == MAX_TAUGHT and "hey helios" not in kept  # the oldest went first
    assert calibrator(tmp_path).promoted()  # read back from disk
    assert item.remove(kept[0]) and kept[0] not in item.promoted()
    assert not item.add_taught(["hey alex"]) or "hey alex" not in item.promoted()  # the guard holds here too


@pytest.mark.asyncio
async def test_a_taught_spelling_wakes_heliox_with_its_punctuation(tmp_path):
    dispatched: list[str] = []
    done = asyncio.Event()

    async def on_command(text: str) -> None:
        dispatched.append(text)
        done.set()

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_command=on_command, config=PilotConfig())
    listener._wake_calibrator = calibrator(tmp_path)
    listener._wake_calibrator.add_taught(["hey helios"])
    said = iter(["Hey, Helios! Open my mail."])

    async def transcribe(**_kwargs):
        try:
            return next(said)
        except StopIteration:
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True
    await asyncio.wait_for(listener._listen_loop(), timeout=5)
    assert dispatched == ["Open my mail"]


@pytest.mark.asyncio
async def test_the_teach_step_listens_dispatches_nothing_and_keeps_what_the_recogniser_wrote(tmp_path):
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener._wake_calibrator = calibrator(tmp_path)
    heard = iter(["Hey Helios.", "Hey Helios!", "Hey Heliox.", "Hey Helios"])
    progress: list[tuple[int, int, str]] = []

    async def capture(_timeout):
        return {"transcript": "ignored", "whisper": next(heard)}

    async def on_progress(n, of, words):
        progress.append((n, of, words))

    listener.capture_diagnostic = capture
    result = await listener.teach_wake(4, on_progress=on_progress)
    assert result["added"] == ["hey helios"] and result["taught"] == ["hey helios"]
    assert result["already_heliox"] == 1 and [p[0] for p in progress] == [1, 2, 3, 4]


def test_a_new_person_s_whisper_is_the_one_on_their_computer(monkeypatch):
    present = {"base"}
    monkeypatch.setattr(model_presence, "whisper_present", lambda size: size in present)
    assert model_presence.usable_whisper_model("small") == "base"  # settings default absent, base downloaded
    present.add("small")
    assert model_presence.usable_whisper_model("small") == "small"
    present.clear()
    assert model_presence.usable_whisper_model("small") == "small"  # nothing: loading says what is missing
    listener = ContinuousVoiceListener(config=PilotConfig())
    present.add("base")
    assert listener._whisper_model() == "base"


@pytest.mark.asyncio
async def test_downloading_whisper_base_switches_a_new_person_off_the_absent_default(monkeypatch):
    from unittest.mock import AsyncMock

    from pilot.server import PilotServer

    present = {"base"}
    monkeypatch.setattr(model_presence, "whisper_present", lambda size: size in present)
    server = PilotServer(PilotConfig())
    server.config.save = lambda *a, **k: None
    server._broadcast_notification = AsyncMock()
    server.config.voice.whisper_model = "small"  # an older settings file
    await server._speech_model_ready("whisper-base")
    assert server.config.voice.whisper_model == "base"
    other = PilotServer(PilotConfig())
    other.config.save = lambda *a, **k: None
    other._broadcast_notification = AsyncMock()
    other.config.voice.whisper_model = "small"
    present.add("small")
    await other._speech_model_ready("whisper-base")
    assert other.config.voice.whisper_model == "small"  # their chosen size is here: left alone


def test_a_near_miss_is_compared_without_its_punctuation(tmp_path):
    assert calibrator(tmp_path).check_near_miss("hey heliocs, open my mail") == "hey heliocs"


@pytest.mark.parametrize(
    ("language", "system_locale", "size"),
    [("auto", "English_United States", "base"), ("auto", "en_GB", "base"), ("en", "de_DE", "base"),
     ("auto", "German_Germany", "small"), ("de", "en_US", "small"), ("auto", "C", "base")],
)  # fmt: skip
def test_the_default_whisper_is_what_the_recommended_download_brings(language, system_locale, size):
    from pilot.system.speech_models import default_whisper_model

    assert default_whisper_model(language, system_locale) == size
