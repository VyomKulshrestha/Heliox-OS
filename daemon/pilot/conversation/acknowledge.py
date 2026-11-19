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

So that it does not become a habit (spaced_out): the two take turns, so the same hum is never heard twice running; it
is never said on the turn after one that had it; on the turn after that it is said half the time; after that, as on
any turn. A run of slow answers then hears one every second or third turn, at no fixed rhythm.

Never: on a typed turn; on a turn whose last word's time is unknown (it cannot be timed); on control words, a turn of
fewer than ACK_MIN_WORDS words, a thanks or an acknowledgement; on a statement (its one-sentence reply is the
acknowledgement); on the turn after one that had it; with conversation.acknowledge off. A turn that had one gets no
holding phrase as well.
"""

from __future__ import annotations

import re
from collections.abc import Callable

ACKNOWLEDGEMENTS = ("Mm.", "Hmm.")
ACK_AFTER_SECONDS = 0.45  # after the person's last word
ACK_MIN_WORDS = 3
# The chance of one on the second turn after a turn that had one (never on the first; always from the third on).
SECOND_TURN_CHANCE = 0.5

_WORDS = re.compile(r"[\w']+")


def wants_acknowledgement(
    text: str,
    *,
    typed: bool,
    ended: float | None,
    statement: bool,
    acked_last_turn: bool,
    enabled: bool,
    language: str = "en",
) -> bool:
    """Whether this spoken turn on the model's road may get an acknowledgement at all (its timing is decided later)."""
    if typed or ended is None or statement or acked_last_turn or not enabled:
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


def spaced_out(turns_since: int | None, draw: Callable[[], float]) -> bool:
    """Whether a turn ``turns_since`` turns after the last one that had an acknowledgement (None: none had one yet) may
    have one: never on the next turn, SECOND_TURN_CHANCE of the time on the one after (``draw`` gives a number in
    [0, 1), asked only then), always from the third on."""
    if turns_since is None or turns_since >= 3:
        return True
    if turns_since <= 1:
        return False
    return draw() < SECOND_TURN_CHANCE
