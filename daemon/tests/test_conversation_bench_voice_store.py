"""The evidence the timing run keeps about the local voice: which voice spoke each sentence, whether its piece waited
behind a fixed sentence being made ahead, the store's counts, and the voice worker's log lines."""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

from benchmarks.conversation_bench import timeline as tl
from benchmarks.conversation_bench.driver import Harness, synth_wait, voice_evidence, voice_marks, voice_replies
from benchmarks.conversation_bench.models import KINDS
from pilot import config
from pilot.system import local_tts_worker as worker


def _harness(tmp_path: Path) -> Harness:
    return Harness(audio_root=str(tmp_path), data_dir=tmp_path / "root", model=KINDS["cloud"])


def test_a_reply_that_waited_behind_a_piece_made_ahead_is_named():
    records = [
        ("ahead", 3, 100.0, 100.0, 101.2),  # a fixed sentence, being made from 100.0 to 101.2
        ("reply", 1, 100.5, 101.2, 101.6),  # "Okay." asked for at 100.5, taken only when the piece was done
    ]
    got = synth_wait(records, 0)

    assert got == {"synth_wait_s": 0.7, "synth_s": 0.4, "behind_ahead_piece": True}
    assert synth_wait(records[1:], 0)["behind_ahead_piece"] is False
    assert synth_wait(records, 2) == {}  # nothing made since the mark


def test_the_voice_that_spoke_is_read_from_the_managers_own_records(monkeypatch):
    manager = worker.manager()
    monkeypatch.setattr(manager, "synth_log", collections.deque(maxlen=200))
    monkeypatch.setattr(manager, "last_reply", None)
    mark = voice_marks()
    record = worker.record_reply_voice("system", "system_still_loading", "the local voice is loading")
    manager.synth_log.append(("reply", 2, 5.0, 5.1, 5.9))

    got = voice_evidence(mark)

    assert got["voice"] == "system" and got["branch"] == "system_still_loading"
    assert got["reason"] == "the local voice is loading" and got["synth_wait_s"] == 0.1
    assert record.sequence > mark[0]
    assert "voice" not in voice_evidence(voice_marks())  # nothing new since


def test_each_sentence_is_grouped_by_voice_and_by_waiting_behind_a_piece():
    def ev(t, kind, **data):
        return tl.Event(t=t, event=kind, detail="", data=data)

    events = [
        ev(1.0, tl.SPEAKS, setup=True),
        ev(1.5, tl.AUDIO_STARTS),
        ev(2.0, tl.SPEAKS, setup=False),
        ev(3.0, tl.AUDIO_STARTS),
        ev(3.5, tl.VOICE_REPLY, voice="local", synth_wait_s=0.7, behind_ahead_piece=True),
        ev(5.0, tl.SPEAKS, setup=False),
        ev(5.2, tl.AUDIO_STARTS),
        ev(5.3, tl.VOICE_REPLY, voice="store", synth_wait_s=None, behind_ahead_piece=False),
    ]
    got = voice_replies(events)

    assert got["voice local"] == {"n": 1, "median_to_audio_s": 1.0, "median_synth_wait_s": 0.7}
    assert got["voice store"]["n"] == 1 and got["voice store"]["median_to_audio_s"] == 0.2
    assert got["waited behind a piece made ahead"]["n"] == 1
    assert got["no piece made ahead in the way"]["n"] == 1 and got["voice system"]["n"] == 0


def test_the_store_is_counted_and_the_worker_log_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    harness = _harness(tmp_path)
    assert harness.voice_store_counts() == {"entries": 0, "fixed": 0, "kept": 0}
    index = tmp_path / "cache" / "voice_heads" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"a": {"fixed": True, "kept": True}, "b": {"kept": True}, "c": {}}), "utf-8")
    assert harness.voice_store_counts() == {"entries": 3, "fixed": 1, "kept": 2}

    loggers = [logging.getLogger(n) for n in ("pilot.system.local_tts_worker", "pilot.system.voice")]
    levels = [lg.level for lg in loggers]
    path = harness.keep_voice_log(tmp_path / "results")
    try:
        loggers[0].info("Released the local voice (%s)", "a test")
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            for handler in [h for h in lg.handlers if getattr(h, "baseFilename", "") == str(path)]:
                lg.removeHandler(handler)
                handler.close()
            lg.setLevel(level)
    assert "Released the local voice (a test)" in path.read_text(encoding="utf-8")
