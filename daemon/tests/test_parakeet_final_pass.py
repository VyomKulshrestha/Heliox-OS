"""Parakeet as the final pass: where its model is looked for, how its words and Whisper's wake phrase combine,
and that nothing changes for someone without the model. Transcripts are from the owner's rehearsal (text only);
sherpa-onnx and the model are never needed here."""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.system import parakeet_asr
from pilot.system.parakeet_asr import FILES, MODEL_NAME, is_filler, merge_wake, missing_files, model_dir
from pilot.system.voice import ContinuousVoiceListener, loose_wake_prefix

PARAKEET = "Dear Yorks remind me that the launch call is Monday, the twelfth of November at ten."
WHISPER = "Here Heliox, remind me that the launch call is mildly the 12th of November at 10th."


def test_parakeets_words_with_whispers_wake_phrase():
    assert merge_wake(WHISPER, PARAKEET, loose_wake_prefix) == (
        "Here Heliox, remind me that the launch call is Monday, the twelfth of November at ten."
    )


def test_when_the_two_do_not_line_up_the_dates_still_come_from_parakeet():
    merged = merge_wake(
        "Heliox, invite me to the launch, pause, Monday, 12th, 11th, 10th.",
        "The area's alignment at the last fall is Monday, the twelfth of November at ten.",
        loose_wake_prefix,
    )
    assert merged.startswith("Heliox, ") and "Monday, the twelfth of November at ten" in merged


def test_without_a_wake_phrase_parakeets_words_are_used():
    assert merge_wake("Skip the back up.", "Skip the backup.", loose_wake_prefix) == "Skip the backup."


@pytest.mark.parametrize("lone", ["Okay.", "Wait."])
def test_a_lone_word_on_noise_whisper_rated_not_speech_is_dropped(lone):
    assert merge_wake("", lone, loose_wake_prefix) == ""


def test_fillers_are_not_words():
    for filler in ("Mm-hmm.", "Uh", "'Kay.", "So", ""):
        assert is_filler(filler)
    assert not is_filler("Yes.") and not is_filler("No")


@pytest.mark.parametrize("pure", [PurePosixPath, PureWindowsPath])
def test_the_model_folder_is_the_setting_or_the_data_folder(pure):
    data = pure("/data/heliox") if pure is PurePosixPath else pure("C:/data/heliox")
    assert model_dir("", Path(str(data))) == Path(str(data)) / "models" / MODEL_NAME
    assert model_dir("D:/models/parakeet", Path(str(data))) == Path("D:/models/parakeet")


def test_missing_files_are_named(tmp_path):
    assert missing_files(tmp_path) == list(FILES)
    for name in FILES:
        (tmp_path / name).write_text("x")
    assert missing_files(tmp_path) == []


def test_the_setting_defaults_to_whisper_and_is_validated():
    assert PilotConfig().voice.final_pass == "whisper" and PilotConfig().voice.parakeet_model_dir == ""
    merged = _merge_config(PilotConfig(), {"voice": {"final_pass": "parakeet", "parakeet_model_dir": "D:/m"}})
    assert merged.voice.final_pass == "parakeet"


def listener(tmp_path, final_pass: str, with_model: bool):
    config = PilotConfig()
    config.voice.final_pass = final_pass
    config.voice.parakeet_model_dir = str(tmp_path)
    if with_model:
        for name in FILES:
            (tmp_path / name).write_text("x")
    item = ContinuousVoiceListener(config=config)
    item._transcribe_whisper_text = AsyncMock(return_value=WHISPER)
    return item


@pytest.mark.asyncio
async def test_without_the_model_whisper_stays_the_final_pass(tmp_path, monkeypatch):
    item = listener(tmp_path, "parakeet", with_model=False)
    assert await item._transcribe_path("utterance.wav") == WHISPER
    status = item._final_pass_status()
    assert status["final_pass"] == "whisper" and "Parakeet model not found" in status["final_pass_notice"]


@pytest.mark.asyncio
async def test_with_the_model_parakeet_runs_beside_whisper(tmp_path, monkeypatch):
    import numpy as np

    item = listener(tmp_path, "parakeet", with_model=True)
    fake = MagicMock()
    fake.transcribe.return_value = PARAKEET
    fake.available = True
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr("pilot.system.voice._load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    text = await item._transcribe_path("utterance.wav")
    assert text.endswith("Monday, the twelfth of November at ten.") and text.startswith("Here Heliox")
    assert item.last_transcription_engine == "parakeet"
    item._transcribe_whisper_text.assert_awaited_once()
    # English, not detected: detection read the owner's "Hey Heliox, what can I say?" as Dutch on a desk microphone.
    assert item._transcribe_whisper_text.await_args.kwargs == {"language": "en"}


def test_loading_needs_sherpa_only_when_the_model_is_used(tmp_path, monkeypatch):
    transcriber = parakeet_asr.ParakeetTranscriber(tmp_path)
    assert transcriber.available is False and transcriber.load() is False and "missing" in transcriber.error
    assert transcriber.transcribe([0.0] * 160) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "whisper_heard", "expected", "whisper_runs"),
    [
        # A reply mid-conversation needs no wake phrase: Parakeet alone.
        ("Skip the backup and the logs.", WHISPER, "Skip the backup and the logs.", False),
        # Not a reply as Parakeet wrote it: Whisper says it began with the wake phrase.
        (
            PARAKEET,
            WHISPER,
            "Here Heliox, remind me that the launch call is Monday, the twelfth of November at ten.",
            True,
        ),
        ("Okay.", "", "", True),  # a lone word on noise: Whisper rates it not speech
        ("Skip it.", "Skip it.", "Skip it.", True),
    ],
)
async def test_mid_conversation_parakeet_runs_alone_and_whisper_checks_a_lone_word(
    tmp_path, monkeypatch, said, whisper_heard, expected, whisper_runs
):
    import asyncio
    import time

    import numpy as np

    item = listener(tmp_path, "parakeet", with_model=True)

    async def slow_whisper(_path, **_kwargs):  # the wake check runs beside Parakeet; a reply must not wait for it
        await asyncio.sleep(0.5)
        return whisper_heard

    item._transcribe_whisper_text = slow_whisper
    fake = MagicMock()
    fake.transcribe.return_value = said
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr("pilot.system.voice._load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    item.arm_follow_up_window(30)
    started = time.perf_counter()
    assert await item._transcribe_path("utterance.wav") == expected
    waited = time.perf_counter() - started
    assert (waited >= 0.45) == whisper_runs  # Whisper's answer was waited for only when it decides something


@pytest.mark.asyncio
async def test_mid_conversation_live_gets_parakeets_words_before_whisper_answers(tmp_path, monkeypatch):
    import asyncio

    import numpy as np

    item = listener(tmp_path, "parakeet", with_model=True)
    item._loop = asyncio.get_running_loop()
    offered: list[tuple[float, str]] = []
    whisper_done: list[float] = []

    async def overheard(text):
        offered.append((asyncio.get_running_loop().time(), text))

    async def slow_whisper(_path, **_kwargs):
        await asyncio.sleep(0.3)
        whisper_done.append(asyncio.get_running_loop().time())
        return WHISPER

    item._on_overheard = overheard
    item._transcribe_whisper_text = slow_whisper
    fake = MagicMock()
    fake.transcribe.return_value = PARAKEET  # "Dear Yorks remind me…": not a reply as written
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr("pilot.system.voice._load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    item.arm_follow_up_window(30)
    await item._transcribe_path("utterance.wav")
    assert offered and offered[0][1] == PARAKEET and offered[0][0] < whisper_done[0]


@pytest.mark.asyncio
async def test_after_the_microphone_button_parakeets_words_are_used_without_waiting(tmp_path, monkeypatch):
    import asyncio
    import time

    import numpy as np

    item = listener(tmp_path, "parakeet", with_model=True)

    async def slow_whisper(_path, **_kwargs):
        await asyncio.sleep(0.5)
        return WHISPER

    item._transcribe_whisper_text = slow_whisper
    fake = MagicMock()
    fake.transcribe.return_value = PARAKEET
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr("pilot.system.voice._load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    item.note_interruption(by_button=True)
    started = time.perf_counter()
    assert await item._transcribe_path("utterance.wav") == PARAKEET
    assert time.perf_counter() - started < 0.4


# ── the wake phrase read from the opening, while the person is still speaking ──


@pytest.mark.parametrize(
    ("opening", "final", "agrees"),
    [
        ("Hey Heliox, what can", "Hey Lokes, what can I say?", True),  # 22:01 desk microphone
        ("Hey Heliox, what can I", "Hey Elioks, what can I say?", True),
        ("Here Heliox, remind me", "Dear Yorks remind me that the launch call is Monday", True),  # headset
        ("Hey Heliox, what's the best", "Hey Nydiok, what can I say to you?", True),  # 22:01, cut mid-sentence
        ("Hey Heliox.", "Hey Lux, what can I say?", False),  # nothing after the phrase yet: check the whole
        ("Hey Heliox, open", "Hey Felix, what time is it?", False),  # the recognisers disagree: check the whole
        ("Hey Felix, what can", "Hey Felix, what can I say?", False),  # no wake phrase at all
        ("", "Hey Lokes, what can I say?", False),
    ],
)
def test_the_opening_counts_only_when_both_recognisers_heard_the_same_start(opening, final, agrees):
    from pilot.system.voice import early_wake_agrees

    assert early_wake_agrees(opening, final) is agrees


def _parakeet_listener(tmp_path, monkeypatch, said: str):
    import numpy as np

    item = listener(tmp_path, "parakeet", with_model=True)
    fake = MagicMock()
    fake.transcribe.return_value = said
    fake.available = True
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr("pilot.system.voice._load_pcm_wav_for_whisper", lambda _path: np.zeros(16000, np.float32))
    return item


@pytest.mark.asyncio
async def test_an_agreeing_opening_answers_without_reading_the_whole_utterance_again(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "Hey Lokes, what can I say?")

    async def opening():
        return "Hey Heliox, what can"

    item._early_wake = asyncio.ensure_future(opening())
    text = await item._transcribe_path("utterance.wav")
    assert text == "Hey Heliox, what can I say?" and item.last_wake_source == "opening"
    item._transcribe_whisper_text.assert_not_awaited()  # no second Whisper pass after the turn ended


@pytest.mark.asyncio
async def test_an_opening_that_does_not_agree_falls_back_to_the_whole_utterance(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "Hey Felix, what time is it?")

    async def opening():
        return "Hey Heliox, open"

    item._early_wake = asyncio.ensure_future(opening())
    item._transcribe_whisper_text = AsyncMock(return_value="Hey Felix, what time is it?")
    text = await item._transcribe_path("utterance.wav")
    assert text == "Hey Felix, what time is it?" and item.last_wake_source == "whole"
    item._transcribe_whisper_text.assert_awaited_once()  # decided by the whole-utterance check, as before


@pytest.mark.asyncio
async def test_the_recorder_hands_over_the_opening_once_while_the_person_speaks():
    import numpy as np

    from pilot.system.voice import _ContinuousRecorder

    rec = _ContinuousRecorder(start_frames=2, silence_frames=6)
    rec._queue = asyncio.Queue()
    rec.opening_frames = 10
    openings = []
    rec.on_opening = lambda frames: openings.append(len(frames))
    loud, quiet = np.full(800, 10000, dtype=np.int16), np.zeros(800, dtype=np.int16)
    for frame in [loud] * 30 + [quiet] * 10:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=3, keep_backlog=True)
    assert openings == [10]
