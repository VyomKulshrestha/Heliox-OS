"""The store of made speech without soundfile (an optional audio package): a piece in memory is simply not kept, and a
hum kept with its whole quiet tail stays as it is until it can be cut, then is cut once.

The import is made to fail here (sys.modules), so the test means the same whether or not the package is installed."""

from __future__ import annotations

import sys

import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import ACK_TAIL_SECONDS, Piece, SpeechStore

SIG = SpeechStore.signature("kokoro_tts", "af_heart")


@pytest.fixture
def no_soundfile(monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", None)  # "import soundfile" raises ImportError


def test_a_piece_in_memory_is_not_kept_and_nothing_raises(tmp_path, no_soundfile):
    store = SpeechStore(tmp_path / "voice_heads")
    piece = Piece("Okay.", audio=[0.1] * 240, rate=24000)
    assert store.add_fixed("Okay.", SIG, piece) is False
    assert not store.has("Okay.", SIG) and store.fixed_bytes(SIG) == 0
    for _ in range(local_tts_worker.HEAD_CACHE_AFTER + 1):
        assert store.said("Okay.", SIG, piece) is False  # counted, never kept, never an error
    assert not store.has("Okay.", SIG)
    assert not list(store.folder.glob("*.wav"))


def test_a_hum_is_marked_cut_only_once_the_cut_has_run(tmp_path, no_soundfile, monkeypatch):
    store = SpeechStore(tmp_path / "voice_heads")
    store.folder.mkdir(parents=True)
    key = SpeechStore.key("Hmm.", SIG)
    (store.folder / f"{key}.wav").write_bytes(b"RIFF-a hum kept with its whole tail")
    store._load()[key] = {"text": "Hmm.", "signature": SIG, "count": 0, "kept": True, "fixed": True}
    store._save()
    assert store.trim_short_tails() == 0
    assert "tail" not in store._load()[key]  # nothing was cut: not marked, so it is cut when it can be
    assert "tail" not in SpeechStore(store.folder)._load()[key]

    cuts: list[str] = []

    def shape(path):
        cuts.append(path.name)
        return 0, 100

    monkeypatch.setattr(local_tts_worker, "shape_hum_file", shape)  # the package is there now
    assert store.trim_short_tails() == 1 and cuts == [f"{key}.wav"]
    assert store._load()[key]["tail"] == ACK_TAIL_SECONDS and store._load()[key]["hum"] == local_tts_worker.HUM_SHAPE
    assert store.trim_short_tails() == 0 and len(cuts) == 1  # once
