"""Words said as Heliox ends a reply are heard, and nothing a person says vanishes without a record.

Batch five's spoken run (24 November, t2gap_b-1): Heliox's first piece ended at 112.35 s and Heliox, hearing the
person, ended its reply there; the person's first word had come at 112.30 s and their voice went on to 113.55 s.
No next piece played, so no barge-in fired; the listener judged the utterance as begun while Heliox was talking
and dropped "Actually, how tall is it?" without a decision, an overheard entry or a log line. Fakes and the
listener's own clock values set by hand; nothing waits on real time; no audio library is imported."""

from __future__ import annotations

import logging

import pytest

from pilot.system.voice import PAST_HELIOX_SECONDS, STRADDLE_SECONDS
from tests.test_voice_conversation import make_listener


def as_heliox_stopped(words: str, *, began_before: float, went_on: float, voiced: float = 1.2):
    """The listener with a conversation open; the utterance's voice began ``began_before`` seconds before Heliox's
    voice stopped and went on ``went_on`` seconds after it, with no barge-in."""
    listener, run = make_listener([], follow_up=30)

    async def transcribe(**_kwargs):
        listener.suppress_wake_free_commands()  # Heliox speaks its first piece while this utterance is captured
        listener.resume_wake_free_commands()  # ...and ends its reply: the run's 112.35 s
        quiet = listener._heliox_quiet_since
        listener._recorder.speech_started_at = quiet - began_before
        listener._recorder.last_turn_ended_at = quiet + went_on
        listener._recorder.last_voiced_seconds = voiced
        listener._running = False
        return words

    listener._record_and_transcribe = transcribe
    return listener, run


@pytest.mark.asyncio
async def test_the_run_s_timing_the_question_is_answered():
    # First word 0.05 s before Heliox's voice stopped (112.30 against 112.35), voice on until 113.55.
    _, run = as_heliox_stopped("Actually, how tall is it?", began_before=0.05, went_on=1.2)
    dispatched, overheard = await run()
    assert dispatched == ["Actually, how tall is it"] and overheard == []


@pytest.mark.asyncio
async def test_a_stop_phrase_said_as_heliox_stops_is_a_stop():
    _, run = as_heliox_stopped("Wait, stop.", began_before=0.1, went_on=0.6, voiced=0.7)
    assert await run() == (["Stop"], [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("began_before", "went_on"),
    [(STRADDLE_SECONDS + 0.5, 1.2), (0.3, PAST_HELIOX_SECONDS - 0.3)],  # began long before; ended with Heliox
)
async def test_what_ends_with_heliox_s_voice_is_its_echo_and_is_recorded_not_dropped(began_before, went_on, caplog):
    listener, run = as_heliox_stopped(
        "Built for the World's Fair in Paris.", began_before=began_before, went_on=went_on
    )
    with caplog.at_level(logging.INFO, logger="pilot.system.voice"):
        dispatched, overheard = await run()
    assert dispatched == []
    assert overheard == ["Built for the World's Fair in Paris."]  # the record: Live hears it as not addressed
    assert any("Not taken: heard while Heliox was talking" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_every_utterance_not_taken_leaves_a_reason_in_the_log(caplog):
    listener, run = make_listener(["The invoice is due on Friday.", "PowerShell"])  # no conversation open
    with caplog.at_level(logging.INFO, logger="pilot.system.voice"):
        assert await run() == ([], [])
    reasons = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Not taken:")]
    assert reasons == ["Not taken: no conversation open", "Not taken: no conversation open"]
    assert all("invoice" not in r and "PowerShell" not in r for r in reasons)  # reasons only, never the words
