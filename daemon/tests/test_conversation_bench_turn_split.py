"""The turn-split measurement feeds the listener's own recorder on the audio clock: a pause shorter than the
endpointer's silence keeps one sentence whole, a long one makes two, an early Smart Turn below its threshold cuts at a
short pause, and the same clip gives the same endpoints every time. Generated tones, no models, no audio files."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from benchmarks.conversation_bench import turn_split  # noqa: E402

RATE = 16000


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), np.int16)


def sentence(pause: float) -> np.ndarray:
    return np.concatenate([gap(0.3), tone(1.0), gap(pause), tone(1.0)])


class SureItEnded:
    """A Smart Turn that is always 0.96 sure the sentence is finished."""

    ready = True

    def __init__(self) -> None:
        self.calls = 0

    def probability(self, _audio):
        self.calls += 1
        return 0.96


def recorder(detector=None):
    from pilot.system.voice import _ContinuousRecorder

    item = _ContinuousRecorder()
    item.use_file_source()
    item.turn_detector = detector
    return item


@pytest.mark.asyncio
async def test_a_short_pause_keeps_the_sentence_whole_and_a_long_one_makes_two():
    item = recorder()
    assert turn_split.pieces(await turn_split.endpoints(item, sentence(0.3))) == 1
    assert turn_split.pieces(await turn_split.endpoints(item, sentence(1.5))) == 2


@pytest.mark.asyncio
async def test_an_early_smart_turn_cuts_at_a_short_pause_only_below_its_threshold():
    detector = SureItEnded()
    clips = [("pause", sentence(0.3))]
    results = await turn_split.run(clips, [1.0, 0.95], lambda: recorder(detector))
    assert results["1.0"][0]["pieces"] == 1  # 0.3 s is less than the 400 ms check: nothing asks
    assert results["0.95"][0]["pieces"] == 2  # asked at 200 ms, 0.96 >= 0.95: the sentence is cut in two
    assert results["0.95"][0]["endpoints"][0]["reason"] == "smart_turn_early"
    rows = {row["threshold"]: row for row in turn_split.summary(results)}
    assert rows["0.95"]["split"] == ["pause"] and rows["1.0"]["split"] == []


@pytest.mark.asyncio
async def test_the_same_clip_gives_the_same_endpoints_every_time():
    clip = sentence(0.3)
    first = await turn_split.endpoints(recorder(SureItEnded()), clip)
    second = await turn_split.endpoints(recorder(SureItEnded()), clip)
    assert first == second and first


def test_a_recording_is_cut_into_utterances_at_long_quiet():
    x = np.concatenate([gap(0.5), tone(1.0), gap(1.2), tone(0.8), gap(0.5)]).astype(np.float32) / 32767
    assert len(turn_split.utterances(x)) == 2
    assert turn_split.utterances(np.zeros(100, np.float32)) == []
