"""The first sound comes from the first clause: a long first sentence is synthesised as a head cut at a natural pause
and the rest, played one after the other; each piece's padding is trimmed, so the first sound starts at once and a
clause seam is a comma's pause. Fake workers and synthetic buffers only: no model, no audio device."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import (
    KEEP_CLAUSE_SECONDS,
    KEEP_LEAD_SECONDS,
    LocalTTSWorkerManager,
    clause_pieces,
    ends_sentence,
    silence_edges,
    split_for_speech,
)


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        (
            "Your next meeting is the launch call at ten, then the design review at two, and you are free after four.",
            [
                "Your next meeting is the launch call at ten,",
                "then the design review at two, and you are free after four.",
            ],
        ),
        (
            "The backup finished without errors and took about four minutes on the external drive.",
            ["The backup finished without errors", "and took about four minutes on the external drive."],
        ),
        # No pause where a head could end: whole, as before.
        (
            "I moved the three screenshots from your desktop into the Pictures folder and renamed them by date.",
            ["I moved the three screenshots from your desktop into the Pictures folder and renamed them by date."],
        ),
        # Short: nothing to gain.
        ("Your backup finished just now, all good.", ["Your backup finished just now, all good."]),
        ("Done. I moved three files!", ["Done.", "I moved three files!"]),
    ],
)
def test_a_long_first_sentence_is_cut_at_its_first_natural_pause(text, pieces):
    assert split_for_speech(text) == pieces
    assert " ".join(split_for_speech(text)) == " ".join(text.split())  # nothing lost or reordered


def test_later_pieces_grow_at_most_twice_so_each_is_ready_before_the_last_has_played():
    sentence = (
        "Okay then, I checked the folder, and there are twelve files from Monday, eight from Tuesday, "
        "four from Wednesday, and none at all from the rest of the week, which is odd for a busy month."
    )
    pieces = clause_pieces(sentence)
    sizes = [len(p.split()) for p in pieces]
    assert len(pieces) >= 3 and sizes[0] <= 9
    assert all(later <= 2 * earlier or later == sizes[-1] for earlier, later in zip(sizes, sizes[1:], strict=False))
    assert " ".join(pieces) == sentence


def test_only_the_first_sentence_is_cut():
    long = "You have three meetings today, the first at nine, then one at noon and one at four in the afternoon."
    pieces = split_for_speech(f"Here is your day. {long}")
    assert pieces == ["Here is your day.", long]


@pytest.mark.parametrize(("piece", "ends"), [("at ten,", False), ("at four.", True), ('"stop."', True), ("and", False)])
def test_a_piece_ends_a_sentence_only_on_its_final_mark(piece, ends):
    assert ends_sentence(piece) is ends


def test_the_padding_is_trimmed_to_a_short_lead_and_a_comma_s_tail():
    rate = 24000
    voice = np.full(rate, 0.2, dtype=np.float32)  # one second of sound
    audio = np.concatenate([np.zeros(int(0.35 * rate)), voice, np.zeros(int(0.5 * rate))]).astype(np.float32)
    start, stop = silence_edges(audio, rate, KEEP_LEAD_SECONDS, KEEP_CLAUSE_SECONDS)
    assert abs(start / rate - (0.35 - KEEP_LEAD_SECONDS)) < 0.002
    assert abs((stop - start) / rate - (KEEP_LEAD_SECONDS + 1.0 + KEEP_CLAUSE_SECONDS)) < 0.002
    _start, whole_end = silence_edges(audio, rate, KEEP_LEAD_SECONDS, None)  # a sentence's end keeps its pause
    assert whole_end == len(audio)
    assert silence_edges(np.zeros(100, dtype=np.float32), rate, 0.03, 0.25) == (0, 100)  # no sound: untouched


class _Worker:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.returncode: int | None = None
        self.stdin = self
        self.stdout = self
        self._payload = b""
        self.closed = False

    def write(self, payload: bytes) -> None:
        self._payload = payload

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def readline(self) -> bytes:
        request = json.loads(self._payload)
        if request.get("op") != "load":
            self.texts.append(Path(request["text_file"]).read_text(encoding="utf-8"))
            Path(request["output_file"]).write_bytes(b"RIFF-test-audio")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_the_head_plays_first_and_each_piece_is_trimmed_for_its_place(monkeypatch):
    worker = _Worker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    trims: list[tuple[str, float, float | None]] = []
    monkeypatch.setattr(
        local_tts_worker, "trim_silence", lambda path, lead, tail: trims.append((Path(path).name, lead, tail))
    )
    played: list[str] = []

    async def play(path: Path) -> None:
        played.append(path.name)

    reply = "Your next meeting is the launch call at ten, then the design review at two, and you are free after four."
    await manager.speak("kokoro_tts", reply, "af_heart", None, play)
    await manager.shutdown()
    assert worker.texts == split_for_speech(reply)  # the head was synthesised first, then the rest
    assert len(played) == 2
    assert [(lead, tail) for _name, lead, tail in trims] == [
        (KEEP_LEAD_SECONDS, KEEP_CLAUSE_SECONDS),  # mid-sentence: a comma's pause after it
        (KEEP_LEAD_SECONDS, None),  # the sentence's end keeps its own pause
    ]
