"""A short sound that says "I heard you, I'm thinking", while a spoken answer from the model is on its way.

A person answering a question makes a small sound before the words come; a voice that is silent for two seconds
sounds like it did not hear. Heliox says one of ACKNOWLEDGEMENTS ACK_AFTER_SECONDS after the person's last word when
nothing of the answer is ready, and never when the answer is (content always wins: an acknowledgement not yet started
is dropped, and one playing is cut by the answer's higher priority). The words commit to nothing: no "okay", "right"
or "sure", which said before "No, I can't do that" would make Heliox contradict itself, and before a request with an
effect would sound like a promise.

The set is provisional: chosen for what a voice like Kokoro can say naturally, to be heard by the owner (the speech
agent's test audio). Until then conversation.acknowledge is off by default; a person can turn it on in Settings, Voice. They are string literals here so the speech store makes their audio
ahead (pilot.system.fixed_speech): saying one costs no synthesis.

Never: on a typed turn; on a turn whose last word's time is unknown (it cannot be timed); on control words, a turn of
fewer than ACK_MIN_WORDS words, a thanks or an acknowledgement; on a statement (its one-sentence reply is the
acknowledgement); on the turn after one that had it; with conversation.acknowledge off. A turn that had one gets no
holding phrase as well.
"""

from __future__ import annotations

import re

ACKNOWLEDGEMENTS = ("Mm.", "Mhm.", "Hmm.", "Let me see.", "One sec.")
ACK_AFTER_SECONDS = 0.45  # after the person's last word
ACK_MIN_WORDS = 3

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
