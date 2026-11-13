"""'Thank you.' right after Heliox answered is an acknowledgement when it has voice in it; the same words from
silence or noise (Whisper's stock output) are not. Voiced seconds as the listener's recorder reports them."""

from __future__ import annotations

import pytest

from pilot.system.addressing import THANKS_VOICE_SECONDS, Addressing, Heard


def after_an_answer(now: float = 100.0) -> Addressing:
    clock = {"t": now}
    rule = Addressing(clock=lambda: clock["t"])
    rule.open_since = now - 5  # the person addressed Heliox a moment ago
    rule.heliox_finished()  # and Heliox has just answered
    return rule


def heard(text: str, voiced: float, *, at: float = 101.0, over_heliox: bool = False) -> Heard:
    return Heard(text=text, started=at, ended=at + 1.0, voiced_seconds=voiced, heliox_was_speaking=over_heliox)


@pytest.mark.parametrize("text", ["Thank you.", "Thank you so much!", "Thanks a lot.", "thank you very much"])
def test_thank_you_with_voice_right_after_an_answer_is_taken(text):
    decision = after_an_answer().decide(heard(text, 0.5))
    assert decision.act and decision.why == "follow-up: control"


@pytest.mark.parametrize("voiced", [0.05, 0.1, 0.2, THANKS_VOICE_SECONDS - 0.01])
def test_thank_you_from_silence_or_noise_is_not(voiced):
    assert not after_an_answer().decide(heard("Thank you.", voiced)).act


def test_only_inside_the_follow_up_window_and_only_those_words():
    assert not after_an_answer().decide(heard("Thank you.", 0.5, at=100.0 + 30)).act  # long after the answer
    assert not after_an_answer().decide(heard("Thanks for watching.", 2.0)).act  # Whisper's other stock line
    assert not Addressing(clock=lambda: 100.0).decide(heard("Thank you.", 0.5)).act  # no conversation open
