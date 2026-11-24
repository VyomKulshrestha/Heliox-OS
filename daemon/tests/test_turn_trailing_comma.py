"""A finished request is never held for "the rest of the sentence" because the recogniser ended it on a comma.

Live measurement of 6ea1149 (24 November, turn e2e-20): Parakeet wrote "Hey Heliox, suggest a quick lunch idea,"
where Whisper had written "...idea.", the unfinished-sentence rule waited 2.5 s for more, and no early request was
started. The rule now reads the words: a request or question whose last word can end one is whole whatever mark
follows it; an opening clause, a statement, a dash or a last word no request ends on still waits. Fakes only."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.system.voice import sounds_unfinished

EXACT = "Hey Heliox, suggest a quick lunch idea,"  # Parakeet's words for the turn, as logged


@pytest.mark.parametrize(
    "command",
    ["suggest a quick lunch idea,", "what time is it in Tokyo,", "open my calendar for tomorrow,",
     "when is my next meeting,", "okay so recommend a film for tonight;"],
)  # fmt: skip
def test_a_whole_request_ending_on_a_comma_is_finished(command):
    assert not sounds_unfinished(command)


@pytest.mark.parametrize(
    "command",
    ["set a timer for", "remind me to,", "when I was at school during exams,", "if it rains tomorrow,",
     "The library extended its opening hours during exams,", "Close the Chrome window -", "email the invoice to,",
     "I want to..."],
)  # fmt: skip
def test_words_that_trail_off_still_wait(command):
    assert sounds_unfinished(command)


@pytest.mark.asyncio
@pytest.mark.parametrize(("said", "waits"), [(EXACT, False), ("Hey Heliox, set a timer for", True)])
async def test_the_listener_dispatches_the_turn_at_once_or_waits_for_the_rest(said, waits):
    from tests.test_voice_conversation import make_listener

    listener, run = make_listener([said])
    listener._finish_sentence = AsyncMock(side_effect=lambda first: first)
    dispatched, _ = await run()
    assert listener._finish_sentence.await_count == (1 if waits else 0)
    assert dispatched == ["set a timer for" if waits else "suggest a quick lunch idea"]


def test_the_early_request_starts_on_the_exact_words():
    from pilot.config import PilotConfig
    from pilot.system.voice import ContinuousVoiceListener

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    proof = {"addressed": "wake", "wake_text": EXACT, "said": EXACT}
    assert listener.early_request_words(EXACT, proof) == ("suggest a quick lunch idea", "wake")
    assert listener.early_request_words("Hey Heliox, set a timer for", proof) is None
