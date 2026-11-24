"""A voice that takes over has its hums at once, the store keeps several voices of each engine, the trace names the
voice, and a standby loads beside the running voice only with room for it and 1 GB to spare.

In a live run on c1beb1e a face brought am_michael: no hum played for the next five replies, because nothing made
the new voice's hums during the conversation and the store had dropped af_heart's when am_michael replaced it.
Made-up managers, stores and machines only: no worker process, no model; every order is read from counted events."""

from __future__ import annotations

import inspect
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import local_tts_worker as ltw
from pilot.system import speech_trace
from pilot.system.local_tts_worker import LocalTTSWorkerManager, SpeechStore

GB = 1024**3
AF_HEART = SpeechStore.signature("kokoro_tts", "af_heart")
AM_MICHAEL = SpeechStore.signature("kokoro_tts", "am_michael")
BF_EMMA = SpeechStore.signature("kokoro_tts", "bf_emma")
ALBA = SpeechStore.signature("pocket_tts", "alba")


# ── The store keeps several voices per engine ──


def _fill(store: SpeechStore, tmp_path, signature: str, texts: list[str]) -> None:
    for number, text in enumerate(texts):
        made = tmp_path / f"{abs(hash((signature, text)))}-{number}.wav"
        with wave.open(str(made), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes((np.full(7200, 0.1) * 32767).astype("<i2").tobytes())
        assert store.add_fixed(text, signature, made)


def test_going_back_to_a_face_s_voice_finds_its_hums(tmp_path):
    store = SpeechStore(tmp_path / "voice_heads")
    for signature in (AF_HEART, AM_MICHAEL, ALBA, BF_EMMA):
        store.keep_voices(signature)
        _fill(store, tmp_path, signature, ["Mm.", "Hmm."])
    store.keep_voices(AF_HEART)  # back to the woman's face, after the man's, Pocket and a voice chosen in Settings
    assert all(store.has(hum, sig) for hum in ("Mm.", "Hmm.") for sig in (AF_HEART, AM_MICHAEL, ALBA, BF_EMMA))
    assert store.lookup("Hmm.", AF_HEART) != store.lookup("Hmm.", AM_MICHAEL)  # each voice only ever its own audio
    recent = json.loads((store.folder / "voices.json").read_text(encoding="utf-8"))["recent"]
    assert recent == [AF_HEART, BF_EMMA, ALBA, AM_MICHAEL]


def test_the_oldest_voice_goes_first_beyond_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ltw, "KEEP_OTHER_VOICES", 2)
    store = SpeechStore(tmp_path / "voice_heads")
    for signature in (AF_HEART, AM_MICHAEL, ALBA, BF_EMMA):
        store.keep_voices(signature)
        _fill(store, tmp_path, signature, ["Mm."])
    store.keep_voices(BF_EMMA)
    assert [store.has("Mm.", sig) for sig in (AF_HEART, AM_MICHAEL, ALBA, BF_EMMA)] == [False, True, True, True]


def test_a_store_from_before_keeps_its_one_voice_per_engine(tmp_path):
    folder = tmp_path / "voice_heads"
    store = SpeechStore(folder)
    store.keep_voices(AF_HEART)
    _fill(store, tmp_path, AF_HEART, ["Mm."])
    (folder / "voices.json").write_text(json.dumps({"kokoro_tts": AF_HEART}), encoding="utf-8")  # the old format
    reopened = SpeechStore(folder)
    reopened.keep_voices(ALBA)
    assert reopened.has("Mm.", AF_HEART)


# ── Making a voice's hums at once ──


class Maker:
    """A loaded manager whose pieces are made by a stand-in: every made piece and every wait is counted."""

    def __init__(self, engine="kokoro_tts", *, busy_polls=0, warm_polls=0):
        self.manager = LocalTTSWorkerManager(spawn=lambda *_: None, clock=lambda: 100.0)
        m = self.manager
        m.process, m.engine, m.loaded = SimpleNamespace(poll=lambda: None), engine, True
        m._last_activity = 100.0  # just used: the old rule waited MAKE_AHEAD_IDLE_SECONDS from here
        self.events: list[str] = []
        self.kept: set[tuple[str, str]] = set()
        m.head_cache = SimpleNamespace(
            has=lambda text, sig: (text, sig) in self.kept,
            add_fixed=lambda text, sig, piece: self.kept.add((text, sig)) or True,
        )
        self.busy_polls, self.warm_polls = busy_polls, warm_polls
        m._busy = 1 if busy_polls else 0

        async def make(engine, text, voice, path, folder, **given):
            assert given["start_voice"] is False and given["rank"] == ltw.PIPE_AHEAD  # never a cold start, lowest rank
            self.events.append(f"made {voice} {text}")
            return SimpleNamespace(path=None)

        async def too_warm_now():
            if self.warm_polls:
                self.warm_polls -= 1
                return True
            return False

        async def sleep(_seconds):
            self.events.append("wait")
            if self.busy_polls:
                self.busy_polls -= 1
                m._busy = 1 if self.busy_polls else 0

        m._make, m.too_warm_now, m._sleep = make, too_warm_now, sleep

    def made(self) -> list[str]:
        return [event for event in self.events if event.startswith("made ")]


@pytest.mark.asyncio
async def test_hums_are_made_at_once_after_use_but_only_between_replies_and_not_on_a_warm_card():
    maker = Maker(busy_polls=2, warm_polls=1)
    made = await maker.manager.make_ahead("kokoro_tts", "am_michael", ["Mm.", "Hmm."], quiet=lambda: True,
                                          battery=lambda: False, rest_factor=0.0, idle_seconds=0.0)  # fmt: skip
    assert made == 2
    # Two waits while a reply is spoken, one while the card is warm; each piece is followed by its rest.
    assert maker.events[:4] == ["wait", "wait", "wait", "made am_michael Mm."]
    assert maker.made() == ["made am_michael Mm.", "made am_michael Hmm."]


@pytest.mark.asyncio
async def test_without_the_new_rule_the_fixed_sentences_still_wait_for_an_idle_voice():
    maker = Maker()
    waits = 0

    async def sleep(_seconds):
        nonlocal waits
        waits += 1
        if waits == 3:
            maker.manager._failed_until = 1.0  # stop the test: the voice was never idle long enough

    maker.manager._sleep = sleep
    assert await maker.manager.make_ahead("kokoro_tts", "af_heart", ["Okay."], quiet=lambda: True) == 0
    assert maker.events == []


@pytest.mark.asyncio
async def test_making_stops_when_the_voice_is_no_longer_the_one_speaking():
    maker = Maker()
    asked = 0

    def until() -> bool:
        nonlocal asked
        asked += 1
        return asked > 1

    made = await maker.manager.make_ahead("kokoro_tts", "am_michael", ["Mm.", "Hmm."], idle_seconds=0.0, until=until,
                                          quiet=lambda: True, battery=lambda: False, rest_factor=0.0)  # fmt: skip
    assert made == 1 and maker.made() == ["made am_michael Mm."]


@pytest.mark.asyncio
async def test_a_voice_that_now_speaks_makes_its_hums_first_then_its_fixed_sentences(monkeypatch):
    maker = Maker()
    monkeypatch.setattr(ltw, "_manager", maker.manager)
    monkeypatch.setattr(ltw, "_configured", ("kokoro_tts", "am_michael"))
    import pilot.system.fixed_speech as fixed_speech

    monkeypatch.setattr(fixed_speech, "gather", lambda: ("Okay.", "Mm.", "One moment."))
    ltw._make_ready_soon(maker.manager, "kokoro_tts", "am_michael")
    await ltw._making_ahead[1]
    assert maker.made() == ["made am_michael Mm.", "made am_michael Hmm.", "made am_michael Okay.",
                            "made am_michael One moment."]  # fmt: skip


@pytest.mark.asyncio
async def test_a_face_bringing_another_voice_of_the_same_engine_makes_its_hums_at_once(monkeypatch):
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(ltw, "_make_ready_soon", lambda manager, engine, voice: asked.append((engine, voice)))
    maker = Maker()
    maker.manager.process_device = maker.manager.device
    monkeypatch.setattr(ltw, "_manager", maker.manager)
    assert ltw._begin_switch("kokoro_tts", "am_michael", "af_heart") == "same"
    assert asked == [("kokoro_tts", "am_michael")]


# ── The standby makes its hums before it takes over ──


@pytest.mark.asyncio
async def test_the_standby_takes_over_only_once_its_hums_are_made(monkeypatch):
    old = Maker()
    old.manager.free_memory = lambda: 8 * GB
    old.manager._spoke_until = -100.0  # quiet for long: a reply boundary
    monkeypatch.setattr(ltw, "_manager", old.manager)
    standby = Maker("pocket_tts")
    standby.manager.process = None  # still loading
    after: list[tuple[str, str]] = []
    monkeypatch.setattr(ltw, "_make_ready_soon", lambda manager, engine, voice: after.append((engine, voice)))
    monkeypatch.setattr(ltw, "_spawn_task", lambda coroutine: coroutine.close())
    monkeypatch.setattr(ltw.LocalTTSWorkerManager, "_start_prewarm", lambda self, engine, voice: None)
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: seen.append((event, data)))
    assert ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=lambda: standby.manager) == "standby"
    assert ltw.finish_handover_if_due() is False  # not loaded yet
    standby.manager.process = SimpleNamespace(poll=lambda: None)
    assert ltw.finish_handover_if_due() is False  # loaded: it starts on its hums first
    await ltw.handover().hums
    assert old.events == [] and standby.made() == ["made alba Mm.", "made alba Hmm."]
    assert ltw.finish_handover_if_due() is True and ltw.manager() is standby.manager
    assert after == [("pocket_tts", "alba")]  # then the rest of its fixed sentences
    events = {event: data for event, data in seen}
    assert events["voice standby"]["voice"] == "alba" and events["voice standby"]["from_voice"] == "af_heart"
    assert events["voice takeover"]["voice"] == "alba" and events["voice takeover"]["from_voice"] == "af_heart"


def test_the_standby_does_not_wait_on_its_hums_for_ever(monkeypatch):
    clock = [0.0]
    standby = Maker("pocket_tts")
    standby.manager._clock = lambda: clock[0]
    current = ltw.Handover("pocket_tts", "alba", "kokoro_tts", "af_heart", standby.manager)
    current.hums = SimpleNamespace(done=lambda: False)  # held up by a warm card
    current.hums_since = 0.0
    assert ltw._hums_done(current) is False
    clock[0] = ltw.HUMS_BEFORE_TAKEOVER_SECONDS
    assert ltw._hums_done(current) is True


# ── The trace names the voice ──


def test_the_trace_names_the_voice_that_made_or_loaded_a_piece():
    source = inspect.getsource(LocalTTSWorkerManager)
    for event in ('"voice loaded", engine=engine, voice=voice', '"stream started", engine=engine, voice=voice'):
        assert event in source
    synthesized = source[source.index('"piece synthesized"') :][:200]
    assert "voice=voice" in synthesized and "engine=engine" in synthesized


# ── Room for two workers ──


@pytest.mark.parametrize(
    ("engine", "device", "gigabytes"),
    [("pocket_tts", "cpu", 2.15), ("kokoro_tts", "cpu", 2.6), ("kokoro_tts", "cuda", 2.9), ("new_engine", "cpu", 2.9)],
)
def test_the_floor_is_the_voice_s_measured_size_and_one_gigabyte_left_over(engine, device, gigabytes):
    assert ltw.standby_floor(engine, device) == pytest.approx(gigabytes * GB, rel=0.001)


@pytest.mark.parametrize(
    ("free_gb", "environ", "device", "engine", "fits"),
    [
        (2.5, {"CUDA_VISIBLE_DEVICES": "-1"}, "auto", "pocket_tts", True),  # Pocket beside Kokoro, 16 GB laptop
        (2.5, {"CUDA_VISIBLE_DEVICES": "-1"}, "auto", "kokoro_tts", False),  # Kokoro needs 2.6 on the processor
        (2.7, {"CUDA_VISIBLE_DEVICES": "-1"}, "auto", "kokoro_tts", True),
        (2.7, {}, "auto", "kokoro_tts", False),  # with a card it may load there: 2.9
        (2.7, {}, "cpu", "kokoro_tts", True),  # set to the processor
        (3.0, {}, "auto", "kokoro_tts", True),
        (2.0, {"CUDA_VISIBLE_DEVICES": "-1"}, "auto", "pocket_tts", False),  # the 8 GB machine: release first
    ],
)
def test_two_workers_only_with_room_for_the_second(free_gb, environ, device, engine, fits):
    old = LocalTTSWorkerManager(spawn=lambda *_: None, device=device)
    old.engine, old.environ = "pocket_tts" if engine == "kokoro_tts" else "kokoro_tts", environ
    old.free_memory = lambda: int(free_gb * GB)
    reason = ltw._two_voices_fit(old, engine)
    assert (reason == "") is fits
    if not fits:
        assert "GB of memory is free" in reason and "1 GB left over" in reason
