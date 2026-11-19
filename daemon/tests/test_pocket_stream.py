"""Pocket TTS as a stream (pilot.system.pocket_stream), on a fake model: no package, no weights."""

import threading

import numpy as np
import pytest

from pilot.system import pocket_stream
from pilot.system.pocket_stream import PocketStream

RATE = 1000  # a small rate keeps the arithmetic readable: 80 samples = one 80 ms chunk


class FakeModel:
    sample_rate = RATE

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts: list[str] = []
        self.made = 0  # chunks the model was asked to make
        self.closed = 0
        self.states: list[str] = []

    def get_state_for_audio_prompt(self, voice):
        self.states.append(voice)
        return {"voice": voice}

    def generate_audio_stream(self, state, text):
        self.texts.append(text)
        try:
            for chunk in self.chunks:
                self.made += 1
                yield chunk
        finally:
            self.closed += 1


def quiet(n=80):
    return np.zeros(n, dtype=np.float32)


def sound(n=80, level=0.5):
    return np.full(n, level, dtype=np.float32)


def engine(chunks):
    model = FakeModel(chunks)
    loads = []

    def loader():
        loads.append(1)
        return model

    return PocketStream(loader), model, loads


def test_the_text_reaches_pocket_with_every_number_as_words():
    stream, model, _loads = engine([sound()])
    list(stream.stream("Siobhan called at 3:15 in March 2026."))
    assert model.texts == ["Siobhan called at three fifteen in March twenty twenty-six."]


def test_the_quiet_lead_is_dropped_inside_the_stream_keeping_a_short_lead():
    # Two quiet chunks, then a chunk whose sound starts 20 samples in.
    first_sound = np.concatenate([quiet(20), sound(60)])
    stream, _model, _loads = engine([quiet(), quiet(), first_sound, sound()])
    chunks = list(stream.stream("Okay."))
    keep = int(pocket_stream.KEEP_LEAD_SECONDS * RATE)
    assert len(chunks[0]) == 60 + keep  # the 160 quiet samples before it are gone, but for the kept lead
    assert np.count_nonzero(chunks[0][:keep]) == 0 and chunks[0][keep] == 0.5
    assert [len(c) for c in chunks[1:]] == [80]


def test_a_stream_without_sound_gives_nothing():
    stream, _model, _loads = engine([quiet(), quiet()])
    assert list(stream.stream("Okay.")) == []


def test_a_stop_takes_effect_within_one_chunk_and_closes_the_model_s_stream():
    stop = threading.Event()
    stream, model, _loads = engine([sound() for _ in range(50)])
    got = []
    for chunk in stream.stream("A long sentence.", stop):
        got.append(chunk)
        if len(got) == 2:
            stop.set()
    assert len(got) == 2
    assert model.made <= 3  # at most the chunk being made when the stop came
    assert model.closed == 1


def test_a_stop_before_the_start_makes_nothing():
    stop = threading.Event()
    stop.set()
    stream, model, _loads = engine([sound()])
    assert list(stream.stream("Okay.", stop)) == []
    assert model.made == 0


def test_the_consumer_closing_the_stream_closes_the_model_s_stream():
    stream, model, _loads = engine([sound() for _ in range(10)])
    chunks = stream.stream("Okay.")
    next(chunks)
    chunks.close()  # the player was cut off
    assert model.closed == 1 and model.made == 1


def test_the_model_and_each_voice_load_once():
    stream, model, loads = engine([sound()])
    stream.prepare()
    list(stream.stream("One."))
    list(stream.stream("Two.", voice="anna"))
    list(stream.stream("Three.", voice="anna"))
    assert loads == [1] and model.states == ["alba", "anna"]
    assert stream.loaded and stream.sample_rate == RATE


def test_empty_text_asks_the_model_for_nothing():
    stream, model, loads = engine([sound()])
    assert list(stream.stream("   ")) == []
    assert loads == [] and model.texts == []


@pytest.mark.parametrize("chunk", [np.full((1, 80), 0.5, dtype=np.float32)])
def test_chunks_come_out_flat_float32(chunk):
    stream, _model, _loads = engine([chunk])
    (out,) = list(stream.stream("Okay."))
    assert out.dtype == np.float32 and out.ndim == 1
