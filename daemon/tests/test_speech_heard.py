"""What of a cut-off sentence the person heard: whole words that had finished playing HEARD_MARGIN_SECONDS before the
voice stopped, reported on SpeechOutcome and by heard_of(text); None for a system voice. Fakes only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import local_tts_worker
from pilot.system.companion_speech import CompanionSpeechCoordinator, heard_part
from pilot.system.local_tts_worker import HEARD_MARGIN_SECONDS, LocalTTSWorkerManager, SpeechProgress

RATE = 24000


def playback(seconds: float, played: float) -> SimpleNamespace:
    return SimpleNamespace(audio=np.zeros(int(seconds * RATE)), sample_rate=RATE, position=int(played * RATE))


def test_whole_pieces_then_whole_words_of_the_cut_piece_with_the_margin():
    progress = SpeechProgress(
        "Your next meeting is at ten, then the design review at two.",
        ["Your next meeting is at ten,", "then the design review at two."],
        durations=[2.0],
        playing=playback(2.0, 1.0),
    )
    assert progress.played() == (3.0, 4.0)
    # 3.0 s played, 0.65 s held back: the first piece (2.0 s) whole, and of the second only "then" had ended
    assert progress.heard_words() == 7
    assert progress.heard_words(margin=0.0) == 8  # without the margin, "the" would be counted too
    assert SpeechProgress("a b", ["a b"], durations=[], playing=playback(1.0, 0.5)).heard_words() == 0


def test_a_sentence_that_played_to_the_end_was_heard_whole():
    progress = SpeechProgress("Okay, done.", ["Okay, done."], finished=True)
    assert progress.heard_words() == 2


def test_the_heard_words_are_the_start_of_the_text_as_given():
    assert heard_part("Hey Heliox, the backup is done.", 3, 6) == "Hey Heliox, the"
    assert heard_part("one two three four", 2, 8) == "one"  # another word count: in proportion, rounded down
    assert heard_part("one two", 0, 2) == ""


class Worker:
    def __init__(self) -> None:
        self.returncode = None
        self.stdin = self.stdout = self
        self._payload = b""
        self.closed = False

    def write(self, payload):
        self._payload = payload

    def flush(self):
        return None

    def close(self):
        return None

    def readline(self):
        request = json.loads(self._payload)
        if request.get("op") != "load":
            Path(request["output_file"]).write_bytes(b"RIFF")
        return b'{"status":"ok"}\n'

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_the_local_voice_records_its_pieces_and_where_it_was_cut(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: Worker())
    played = []

    async def play(path: Path) -> None:  # what voice._play_local_tts_file does with the progress record
        progress = local_tts_worker.speech_progress.get()
        progress.playing = playback(2.0, 2.0 if not played else 1.5)
        played.append(path)
        if len(played) == 2:
            raise asyncio.CancelledError  # the person talked over the second piece

    with pytest.raises(asyncio.CancelledError):
        await manager.speak(
            "kokoro_tts",
            "Your next meeting is the launch call at ten, then the design review at two.",
            "af_heart",
            None,
            play,
        )
    progress = manager.last_progress
    assert progress.durations == [2.0] and progress.pieces[0] == "Your next meeting is the launch call at ten,"
    assert progress.heard_words() == 9 + 2  # the first piece, then "then the" by 0.85 s of the second
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_coordinator_reports_what_was_heard_and_heard_of_matches_the_text(monkeypatch):
    fake = SimpleNamespace(last_progress=None)
    monkeypatch.setattr(local_tts_worker, "manager", lambda: fake)
    text = "Your next meeting is at ten, then the design review at two."

    async def local_voice(said: str, recorder) -> bool:
        fake.last_progress = SpeechProgress(
            said, ["Your next meeting is at ten,", "then the design review at two."], sequence=1,
            durations=[2.0], playing=playback(2.0, 1.0),
        )  # fmt: skip
        return True  # interrupted

    coordinator = CompanionSpeechCoordinator(speaker=local_voice)
    outcome = await coordinator.speak(text)
    assert outcome.status == "interrupted"
    assert outcome.heard_text == "Your next meeting is at ten, then"
    assert (outcome.played_seconds, outcome.duration_seconds) == (3.0, 4.0)
    assert coordinator.heard_of(text) == "Your next meeting is at ten, then"
    assert coordinator.heard_of("Something else.") is None
    await coordinator.close()


@pytest.mark.asyncio
async def test_a_system_voice_reports_nothing(monkeypatch):
    fake = SimpleNamespace(last_progress=SpeechProgress("old", ["old"], sequence=4, finished=True))
    monkeypatch.setattr(local_tts_worker, "manager", lambda: fake)

    async def system_voice(said: str, recorder) -> bool:
        return True  # SAPI, say or espeak-ng: a process killed at the cut, no progress record

    coordinator = CompanionSpeechCoordinator(speaker=system_voice)
    outcome = await coordinator.speak("The backup is done and the files are in Downloads.")
    assert outcome.status == "interrupted" and outcome.heard_text is None and outcome.played_seconds is None
    assert coordinator.heard_of("The backup is done and the files are in Downloads.") is None
    await coordinator.close()


def test_the_margin_is_the_longer_end_of_first_sound_to_silence():
    assert HEARD_MARGIN_SECONDS == 0.65
