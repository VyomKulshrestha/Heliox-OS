"""A short sound that says "I heard you, I'm thinking", while a spoken answer from the model is on its way.

A person answering a question makes a small sound before the words come; a voice that is silent for two seconds
sounds like it did not hear. Heliox says one of ACKNOWLEDGEMENTS ACK_AFTER_SECONDS after the person's last word when
nothing of the answer is ready, and never when the answer is (content always wins: an acknowledgement not yet started
is dropped, and one playing is cut by the answer's higher priority). The words commit to nothing: no "okay", "right"
or "sure", which said before "No, I can't do that" would make Heliox contradict itself, and before a request with an
effect would sound like a promise.

The set is two hums, "Mm." and "Hmm.". Four short phrases ("Let me see.", "One sec.", "One moment.", "Let me
check.") came back word for word from Whisper in Heliox's voice on 19 November, but heard in that voice the hums
sounded like a person thinking and the phrases did not. "Mhm." stays out: Kokoro gives it the sound of "Hmm.".
conversation.acknowledge is on by default. They are string literals here so the speech store makes their
audio ahead (pilot.system.fixed_speech), and a hum is said only from that audio (the layer's ``ready_made``): with
the system voice, or before the local voice has made it, nothing is said and nothing waits for it.

So that it does not become a habit (hum_allowed): the two take turns, so the same hum is never heard twice running;
after one turn with a hum the next has one AFTER_ONE_CHANCE of the time, after two in a row AFTER_TWO_CHANCE of the
time, and never a fourth in a row. A turn without one (a skip, or an answer that was ready) starts the count again.
The rule was written for answers that mostly arrive in time; in the 24 November live run none arrived before 1.38 s,
and the old spacing (never on the next turn) left 15 of 24 late turns silent until about 2.1 s. Over a run of late
answers this one hums on about 69 % of turns ((1 + 0.8 + 0.8 * 0.5) / (2 + 0.8 + 0.8 * 0.5)), with the quiet turns at
no fixed place: never a fixed two-on-one-off beat, never the same sound twice running, never four in a row.

Never: on a typed turn; on a turn whose last word's time is unknown (it cannot be timed); on control words, a turn of
fewer than ACK_MIN_WORDS words, a thanks or an acknowledgement; on a statement (its one-sentence reply is the
acknowledgement); when the answer's first text is already there; with conversation.acknowledge off. A turn that had
one gets no holding phrase as well.
"""

from __future__ import annotations

import re
from collections.abc import Callable

ACKNOWLEDGEMENTS = ("Mm.", "Hmm.")
ACK_AFTER_SECONDS = 0.45  # after the person's last word
ACK_MIN_WORDS = 3
# The chance of a hum on a turn right after one turn with a hum, and right after two in a row; never after three.
AFTER_ONE_CHANCE = 0.8
AFTER_TWO_CHANCE = 0.5
MAX_RUN = 3

_WORDS = re.compile(r"[\w']+")


def wants_acknowledgement(
    text: str,
    *,
    typed: bool,
    ended: float | None,
    statement: bool,
    enabled: bool,
    language: str = "en",
) -> bool:
    """Whether this spoken turn on the model's road may get an acknowledgement at all (its timing is decided later)."""
    if typed or ended is None or statement or not enabled:
        return False
    words = _WORDS.findall(text.casefold())
    if len(words) < ACK_MIN_WORDS:
        return False
    from pilot.conversation.layer import BARE_ACK, THANKS
    from pilot.conversation.router import normalize
    from pilot.system import voice_words

    said = normalize(text)
    if THANKS.match(said) or BARE_ACK.match(said):
        return False
    try:
        if voice_words.is_control_word(said, language):
            return False
    except Exception:
        return False
    return True


def next_acknowledgement(last: str) -> str:
    """The next one in turn, never the one said last."""
    if last in ACKNOWLEDGEMENTS:
        return ACKNOWLEDGEMENTS[(ACKNOWLEDGEMENTS.index(last) + 1) % len(ACKNOWLEDGEMENTS)]
    return ACKNOWLEDGEMENTS[0]


def hum_allowed(run: int, draw: Callable[[], float]) -> bool:
    """Whether a late turn may hum, after ``run`` turns in a row that had one: always after none, AFTER_ONE_CHANCE of
    the time after one, AFTER_TWO_CHANCE after two (``draw`` gives a number in [0, 1), asked only then), never after
    MAX_RUN."""
    if run <= 0:
        return True
    if run >= MAX_RUN:
        return False
    return draw() < (AFTER_ONE_CHANCE if run == 1 else AFTER_TWO_CHANCE)
