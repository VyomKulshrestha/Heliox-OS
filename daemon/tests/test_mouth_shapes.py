"""Phoneme timing to mouth shapes, and the loudness envelope used when a voice gives no phoneme timing."""

from __future__ import annotations

import numpy as np

from pilot.system import mouth_shapes
from pilot.system.mouth_shapes import SHAPES, shape_of, timeline


def test_fifteen_shapes_with_silence_and_every_listed_symbol_lands_in_one():
    assert len(SHAPES) == 15 and SHAPES[0] == "sil"
    assert set(mouth_shapes.SHAPE_OF.values()) <= set(SHAPES)


def test_symbols_map_by_how_the_mouth_moves_not_by_language():
    assert [shape_of(s) for s in "pbm"] == ["PP"] * 3
    assert [shape_of(s) for s in "fv"] == ["FF"] * 2
    assert shape_of("θ") == shape_of("ð") == "TH"
    assert shape_of("ʃ") == shape_of("ʧ") == "CH"
    assert shape_of("s") == "SS" and shape_of("n") == "nn" and shape_of("ɹ") == "RR"
    assert shape_of("a") == shape_of("ɑ") == "aa"
    assert shape_of("i") == shape_of("ɪ") == "I"
    assert shape_of("u") == shape_of("w") == "U"
    assert shape_of("o") == shape_of("O") == "O"  # O is Kokoro's English oʊ
    assert shape_of("I") == "aa"  # aɪ starts open
    assert shape_of(" ") == shape_of(".") == shape_of(",") == "sil"
    assert shape_of("ж") == mouth_shapes.UNKNOWN_SHAPE  # never listed: a neutral half-open mouth


def test_a_timeline_has_no_gaps_and_no_two_neighbours_alike():
    out = timeline([["m", 0.1, 0.2], ["a", 0.2, 0.4], ["ɑ", 0.4, 0.5], ["p", 0.6, 0.7]], length=1.0)
    assert out == [["sil", 0, 100], ["PP", 100, 200], ["aa", 200, 500], ["sil", 500, 600], ["PP", 600, 700],
                   ["sil", 700, 1000]]  # fmt: skip
    for before, after in zip(out, out[1:], strict=False):
        assert before[2] == after[1] and before[0] != after[0]


def test_a_short_gap_between_words_keeps_the_mouth_and_a_long_one_closes_it():
    short = timeline([["a", 0.0, 0.2], [" ", 0.2, 0.25], ["o", 0.25, 0.4]])
    assert [s[0] for s in short] == ["aa", "O"] and short[0][2] == 250
    long = timeline([["a", 0.0, 0.2], [" ", 0.2, 0.4], ["o", 0.4, 0.6]])
    assert [s[0] for s in long] == ["aa", "sil", "O"]


def test_stress_and_length_marks_give_their_time_to_their_sound():
    out = timeline([["ˈ", 0.0, 0.05], ["a", 0.05, 0.2], ["ː", 0.2, 0.3], ["t", 0.3, 0.4]])
    assert out == [["aa", 0, 300], ["DD", 300, 400]]


def test_slivers_join_the_shape_before_them():
    out = timeline([["a", 0.0, 0.2], ["t", 0.2, 0.21], ["o", 0.21, 0.4]])
    assert [s[0] for s in out] == ["aa", "O"]
    assert out[0][2] == 210


def test_times_are_clipped_to_the_audio_played():
    out = timeline([["a", -0.1, 0.2], ["o", 0.2, 0.9]], length=0.5)
    assert out == [["aa", 0, 200], ["O", 200, 500]]
    assert timeline([["a", 0.6, 0.9]], length=0.5) == []
    assert timeline([["bad"], "x", [None, "a", "b"]]) == []


def test_the_envelope_is_25_small_integers_a_second_and_zero_in_silence():
    rate = 24000
    quiet = np.zeros(rate // 2, dtype=np.float32)
    loud = 0.3 * np.sin(np.linspace(0, 2000, rate // 2)).astype(np.float32)
    values = mouth_shapes.envelope(np.concatenate([quiet, loud]), rate)
    assert len(values) == 25
    assert all(isinstance(v, int) and 0 <= v <= mouth_shapes.ENVELOPE_TOP for v in values)
    assert max(values[:12]) == 0 and min(values[13:]) >= 12
    assert mouth_shapes.envelope(np.zeros(0), rate) == []
    assert mouth_shapes.envelope(np.zeros(rate), rate) == [0] * 25


def test_a_quiet_voice_still_opens_the_mouth():
    rate = 24000
    soft = 0.01 * np.sin(np.linspace(0, 2000, rate)).astype(np.float32)
    assert max(mouth_shapes.envelope(soft, rate)) >= 12
