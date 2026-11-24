"""A sentence prepared for a reply that was cut off is never made later behind the next reply's pieces.

The fake worker answers each request only when the test lets it; order is checked by counting requests, never by
reading a clock."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_voice_first_piece_first import GatedWorker, loaded_manager, until


async def _nothing(piece) -> None:
    return None


@pytest.mark.asyncio
async def test_a_sentence_prepared_for_a_cut_reply_is_cancelled_and_never_made():
    worker = GatedWorker()
    manager = await loaded_manager(worker)
    first, second = "The first sentence of the reply,", "and the second one, never heard."
    manager.prepare("kokoro_tts", second, "af_heart")  # as the conversation does: the next one, then this one
    speaking = asyncio.ensure_future(manager.speak("kokoro_tts", first, "af_heart", None, _nothing))
    await until(lambda: worker.asked == [first])
    speaking.cancel()  # the person cut the reply off while its first sentence was being made
    with pytest.raises(asyncio.CancelledError):
        await speaking
    worker.gate(first).set()  # the worker finishes the cut sentence unheard

    worker.gate("A new answer.").set()
    await manager.speak("kokoro_tts", "A new answer.", "af_heart", None, _nothing)
    for _ in range(50):
        await asyncio.sleep(0)
    assert worker.asked == [first, "A new answer."]  # the cut reply's second sentence was never made
    assert not manager._prepared and not manager._next_state
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_sentence_prepared_after_the_cut_speech_began_is_kept():
    """The answer's next sentence is prepared just before its first one cuts a hum: cutting the hum keeps it."""
    worker = GatedWorker()
    manager = await loaded_manager(worker)
    humming = asyncio.ensure_future(manager.speak("kokoro_tts", "Hmm.", "af_heart", None, _nothing))
    await until(lambda: worker.asked == ["Hmm."])
    manager.prepare("kokoro_tts", "The answer's second sentence.", "af_heart")
    humming.cancel()
    with pytest.raises(asyncio.CancelledError):
        await humming
    worker.gate("Hmm.").set()
    assert ("kokoro_tts", "af_heart", "The answer's second sentence.") in manager._prepared

    worker.gate("The answer's first sentence.").set()
    worker.gate("The answer's second sentence.").set()
    await manager.speak("kokoro_tts", "The answer's first sentence.", "af_heart", None, _nothing)
    await manager.speak("kokoro_tts", "The answer's second sentence.", "af_heart", None, _nothing)
    assert worker.asked == ["Hmm.", "The answer's first sentence.", "The answer's second sentence."]  # made once
    await manager.shutdown()
