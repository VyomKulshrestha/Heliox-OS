"""A reply the person cut off is remembered as far as they heard it, marked as cut off.

In the companion acceptance run (24 November) the conversation held only the person's question after each of 4
interruptions. Two causes, both fixed here. When the next utterance cancelled the reply mid-sentence, the layer asked
the voice what was heard before the voice had worked it out, and got "unknown" (then the whole sentence, more than was
heard) or the answer for an earlier sentence; a sentence withdrawn before it played, or one whose numbers the voice
said its own way ("1889"), had the same fault. And the bench's silent player never reported how far a piece had
played, so every cut read as nothing heard. Fakes only: a scripted model, a speaker that blocks, the local voice's
progress set by hand; no audio and no real waiting."""

from __future__ import annotations

import asyncio

import pytest

from pilot.conversation.context import ContextSources
from pilot.conversation.layer import ConversationLayer
from pilot.system import local_tts_worker
from pilot.system.companion_speech import CompanionSpeechCoordinator
from pilot.system.local_tts_worker import SpeechProgress

FIRST = "It was built for the World's Fair in Paris."
SECOND = "It is about three hundred metres tall, with three levels open to visitors."


class _Played:
    """A piece's playback, 4 s long, of which ``seconds`` have played."""

    complete = True

    def __init__(self, seconds: float) -> None:
        self.sample_rate = 1000
        self.audio = [0] * 4000
        self.position = int(seconds * 1000)


class _Voice:
    """The local voice manager as the coordinator reads it."""

    def __init__(self) -> None:
        self.last_progress: SpeechProgress | None = None
        self.last_reply = None
        self.count = 0


@pytest.fixture
def voice(monkeypatch):
    found = _Voice()
    monkeypatch.setattr(local_tts_worker, "manager", lambda: found)
    return found


def coordinator_for(voice: _Voice, *, cut: str, played: float) -> tuple[CompanionSpeechCoordinator, asyncio.Event]:
    """A coordinator whose speaker says every sentence at once, except ``cut``, which plays ``played`` seconds and
    then waits until it is stopped."""
    reached = asyncio.Event()

    async def speaker(text: str, _recorder) -> bool:
        voice.count += 1
        progress = SpeechProgress(text, [text], voice.count)
        voice.last_progress = progress
        if text != cut:
            progress.finished = True
            return False
        progress.playing = _Played(played)
        reached.set()
        await asyncio.Event().wait()  # until the voice is stopped
        return False

    return CompanionSpeechCoordinator(speaker=speaker), reached


def layer_on(coordinator: CompanionSpeechCoordinator, fragments: list[str]) -> ConversationLayer:
    class Model:
        name = provider = "fake"
        streams, local = True, False

        async def stream(self, _system, _messages):
            for fragment in fragments:
                yield fragment

    model = Model()

    async def sink(text: str, channel: str, key: str) -> str:
        return (await coordinator.speak(text, channel=channel, dedupe_key=key)).status

    def heard_of(text: str) -> str:  # as pilot.conversation.adapters._heard_of: unknown means the whole sentence
        heard = coordinator.heard_of(text)
        return text if heard is None else heard

    return ConversationLayer(sink=sink, model=lambda: model, sources=ContextSources(heard_of=heard_of))


@pytest.mark.asyncio
async def test_a_reply_cut_by_the_next_utterance_keeps_what_was_heard(voice):
    coordinator, reached = coordinator_for(voice, cut=SECOND, played=2.0)
    layer = layer_on(coordinator, [f"{FIRST} ", f"{SECOND} "])
    turn = asyncio.ensure_future(layer.take("Tell me the story of the Eiffel Tower.", source="voice"))
    await reached.wait()
    assert layer.interrupt()  # "Actually, how tall is it?" arrived: the server cuts the reply off
    await turn
    kept = [t for t in layer.memory.recent() if t.role == "assistant"]
    assert len(kept) == 1 and kept[0].cut_off
    heard = kept[0].text[len(FIRST) :].strip()
    assert kept[0].text.startswith(FIRST) and heard and SECOND.startswith(heard) and heard != SECOND


@pytest.mark.asyncio
async def test_a_sentence_withdrawn_before_it_played_is_not_kept_as_said(voice):
    coordinator, reached = coordinator_for(voice, cut=FIRST, played=0.3)
    held = asyncio.ensure_future(coordinator.speak(FIRST))  # the voice is busy with another line
    await reached.wait()
    queued = asyncio.ensure_future(coordinator.speak(SECOND))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert coordinator.heard_of(SECOND) == ""  # none of it was heard, and that is known
    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held
    assert coordinator.heard_of(FIRST) == ""  # 0.3 s played: within the margin, no whole word heard yet


@pytest.mark.asyncio
async def test_numbers_the_voice_said_its_own_way_still_match(voice):
    from pilot.conversation.style import speakable_numbers

    written = "The report is due at 9 a.m. tomorrow, so it makes sense to start on it tonight after dinner."
    spoken = speakable_numbers(written)
    assert spoken != written
    coordinator, reached = coordinator_for(voice, cut=spoken, played=3.0)
    line = asyncio.ensure_future(coordinator.speak(spoken))  # the server says numbers before the voice gets it
    await reached.wait()
    line.cancel()
    with pytest.raises(asyncio.CancelledError):
        await line
    heard = coordinator.heard_of(written)
    assert heard and written.startswith(heard) and heard != written


@pytest.mark.asyncio
async def test_the_bench_s_silent_player_reports_how_far_a_piece_played(tmp_path):
    from benchmarks.conversation_bench import driver, wavio

    rate = 16000
    path = tmp_path / "piece.wav"
    wavio.write(path, [0.0] * rate, rate)
    raw, read_rate = wavio.decode(path.read_bytes())
    playback = driver._SilentPlayback(raw, read_rate)
    playback._started -= 0.5  # half a second ago, on the clock it reads
    assert 0.45 * rate <= playback.position <= rate
    playback._started -= 5.0
    assert playback.position == len(raw)
