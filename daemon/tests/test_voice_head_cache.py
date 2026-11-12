"""The cache of the heads this Heliox says most. It plays a head only when the piece about to be spoken is exactly that
text, made the same way: it never adds, drops or changes a word of a reply. Fake workers only: no model, no audio
device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import HEAD_CACHE_ENTRIES, HeadCache, LocalTTSWorkerManager, split_for_speech


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        # A one-to-three-word opener is not cut off: on a processor the next piece is not ready when it ends, so
        # "Okay," was followed by 1.9 to 3.6 s of silence (docs/INTERACTION_MODEL.md, the seam table).
        ("Okay, the backup is done.", ["Okay, the backup is done."]),
        ("Sure, I'll remind you at four.", ["Sure, I'll remind you at four."]),
        (
            "Okay, your next meeting is at ten, then the design review at two.",
            ["Okay, your next meeting is at ten,", "then the design review at two."],
        ),
        ("Okay, done. Your files are in Downloads.", ["Okay, done.", "Your files are in Downloads."]),
    ],
)
def test_an_opener_stays_with_its_sentence(text, pieces):
    assert split_for_speech(text) == pieces


@pytest.mark.parametrize(
    "text",
    [
        "Okay, your next meeting is the launch call at ten, then the design review at two, and you are free after four.",
        "Well, it's 4:30 p.m. in Tokyo right now, which is nine hours ahead of London.",
        "Hmm, I couldn't find that file, but there is one with a similar name in Downloads.",
        "Right. Sure, I'll do that. And then, if you want, I can send it.",
    ],
)
def test_the_pieces_are_exactly_the_reply_s_words(text):
    assert " ".join(split_for_speech(text)) == " ".join(text.split())


def wav(path: Path, data: bytes = b"RIFF-head") -> Path:
    path.write_bytes(data)
    return path


def test_a_head_is_kept_after_it_has_been_said_twice_and_found_only_by_its_exact_text(tmp_path):
    cache = HeadCache(tmp_path / "voice_heads")
    sig = HeadCache.signature("kokoro_tts", "af_heart")
    made = wav(tmp_path / "okay.wav")
    assert cache.said("Okay,", sig, made) is False and cache.lookup("Okay,", sig) is None
    assert cache.said("Okay,", sig, made) is True
    kept = cache.lookup("Okay,", sig)
    assert kept is not None and kept.read_bytes() == b"RIFF-head" and kept.parent == tmp_path / "voice_heads"
    for other in ("Okay", "okay,", "Okay, ", "Okay then,"):
        assert cache.lookup(other, sig) is None
    assert cache.lookup("Okay,", HeadCache.signature("kokoro_tts", "am_michael")) is None  # another voice
    assert HeadCache(tmp_path / "voice_heads").lookup("Okay,", sig) is not None  # survives a restart


def test_another_voice_drops_the_kept_heads_and_the_cache_stays_bounded(tmp_path):
    cache = HeadCache(tmp_path / "voice_heads")
    sig = HeadCache.signature("kokoro_tts", "af_heart")
    for n in range(HEAD_CACHE_ENTRIES + 5):
        made = wav(tmp_path / f"h{n}.wav")
        cache.said(f"Head {n},", sig, made)
        cache.said(f"Head {n},", sig, made)
    files = list((tmp_path / "voice_heads").glob("*.wav"))
    assert len(files) == HEAD_CACHE_ENTRIES
    cache.keep_only(HeadCache.signature("kokoro_tts", "am_michael"))
    assert list((tmp_path / "voice_heads").glob("*.wav")) == []
    assert json.loads((tmp_path / "voice_heads" / "index.json").read_text(encoding="utf-8")) == {}


def test_a_long_head_is_never_kept(tmp_path):
    cache = HeadCache(tmp_path / "voice_heads")
    sig = HeadCache.signature("kokoro_tts", "af_heart")
    made = wav(tmp_path / "long.wav")
    text = "Your next meeting is the launch call at ten,"
    assert not cache.said(text, sig, made) and not cache.said(text, sig, made)
    assert cache.lookup(text, sig) is None


class Worker:
    def __init__(self) -> None:
        self.texts: list[str] = []
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
        if request.get("op") != "load":
            text = Path(request["text_file"]).read_text(encoding="utf-8")
            self.texts.append(text)
            Path(request["output_file"]).write_bytes(f"RIFF-{text}".encode())
        return b'{"status":"ok"}\n'

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


async def say(manager, text):
    played: list[bytes] = []

    async def play(path: Path) -> None:
        played.append(path.read_bytes())

    await manager.speak("kokoro_tts", text, "af_heart", None, play)
    return played


@pytest.mark.asyncio
async def test_a_kept_head_plays_without_synthesis_and_never_changes_the_words(monkeypatch, tmp_path):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker = Worker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.head_cache = HeadCache(tmp_path / "voice_heads")
    await say(manager, "Okay, done.")
    await say(manager, "Okay, done.")  # said twice: "Okay, done." is kept
    worker.texts.clear()
    played = await say(manager, "Okay, done.")
    assert worker.texts == [] and played == [b"RIFF-Okay, done."]  # played as kept, nothing synthesised
    played = await say(manager, "Okay, done. Your files are in Downloads.")
    assert worker.texts == ["Your files are in Downloads."]  # the head came from the cache, not the voice
    assert played == [b"RIFF-Okay, done.", b"RIFF-Your files are in Downloads."]
    # A reply that only starts like a kept head is made in full: the cache never puts its words into it.
    for text in ("Okay, done then.", "Okay, done", "Okay, done!"):
        worker.texts.clear()
        played = await say(manager, text)
        assert worker.texts == [text]
        assert b"".join(played).replace(b"RIFF-", b" ").split() == text.encode().split()
    await manager.shutdown()
