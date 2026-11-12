"""When the fast model's reply commits Heliox to doing something, and when it holds back.

The fast model hands a request over with an ACT line. Sometimes it says in words that it will act instead ("I'll get
that message to Priya now.") and writes no ACT line: Heliox must not leave a spoken promise with nothing done, so
the request is handed over. Sometimes it declines or hedges ("I can't tell which files are the old installers from
here, so I won't delete anything yet.") and, in the same reply, promises or writes an ACT line anyway. The decline
wins: nothing is handed over, so a vague request (often a destructive one) never reaches the planner on a guess.

Pure functions, tested on their own; the layer calls them per sentence.
"""

from __future__ import annotations

import re

_TAKING_IT_ON = r"(?:i'll|i will|i'm going to|i am going to)"
_OPENER = rf"\b(?:{_TAKING_IT_ON}|let me|i can do that now and)\s+(?:go ahead and\s+|just\s+|now\s+)?"

# Verbs that are an action on the computer whatever follows them.
_ACTION_VERBS = (
    r"open|close|delete|move|copy|send|launch|start|set|create|turn|mute|unmute|play|pause|search|find|install|"
    r"uninstall|remove|rename|run|schedule|book|empty|switch|lock|restart|shut|save|download|message|email|e-mail|"
    r"remind|zip|unzip|archive|forward|upload|print|notify|organi[sz]e"
)
_PROMISE = re.compile(_OPENER + rf"(?P<verb>{_ACTION_VERBS})\b", re.IGNORECASE)

# Verbs that are also ordinary talk ("let me put it another way", "I'll add one thing"): a promise only after
# "I'll" and with somewhere for the thing to go ("add it to your list") or someone to write to ("text her").
_WEAK_PLACE = re.compile(
    rf"\b{_TAKING_IT_ON}\s+(?:go ahead and\s+|just\s+|now\s+)?"
    r"(?P<verb>add|write|draft|put|share|clear|clean|tidy|sort|post|back)\b"
    r"(?:\s+\S+){0,5}?\s+(?:to|into|in|on|onto|for|up)\s+(?:your|the|my|her|his|their|a|[A-Z])",
    re.IGNORECASE,
)
_WEAK_PERSON = re.compile(
    rf"\b{_TAKING_IT_ON}\s+(?:go ahead and\s+|just\s+|now\s+)?(?P<verb>write|draft|text|call|ping|reply to)\s+"
    r"(?!you\b|me\b)(?:her|him|them|[A-Z]\w+|a (?:note|message|reminder|email|reply))",
    re.IGNORECASE,
)
# "I'll get that message to Priya", "I'll get it done": get with something to deliver or finish.
_GET = re.compile(
    _OPENER + r"(?P<verb>get)\s+(?!you\b|back\b)(?:\S+\s+){0,4}?(?:to\s+(?!you\b)\w|done\b|sent\b|ready\b|going\b)",
    re.IGNORECASE,
)
# "I'll tell her", "I'll let Priya know": telling someone other than the person.
_TELL = re.compile(
    _OPENER + r"(?P<verb>tell|let)\s+(?!you\b|me\b|us\b)(?:her|him|them|your\s+\w+|[A-Z]\w+)\b",
    re.IGNORECASE,
)
_PATTERNS = (_PROMISE, _WEAK_PLACE, _WEAK_PERSON, _GET, _TELL)

# Holding back: not sure what is meant, cannot see it, will not act yet, or acting only after the person answers.
_HEDGE = re.compile(
    r"\b(?:won't|will not|wouldn't|would not|not going to|rather not|shouldn't|should not|unable to|not able to|"
    r"not sure|unsure|no way (?:for me )?to|don't know|do not know|didn't catch|did not catch|"
    r"(?:can't|cannot|can not|couldn't|could not)\s+(?:\w+\s+){0,2}?(?:tell|see|find|know|be sure|work out|figure|"
    r"make out|do that|do this|help with that|reach|access|read)|"
    r"which (?:one|ones|file|files|folder|folders|of them|of those|app|window|message|person)\b|"
    r"(?:once|until|after|when) you (?:tell|confirm|say|let|show|point|pick|choose|name)|"
    r"before i (?:do|delete|move|send|remove|change|touch)|"
    r"(?:tell|show) me which|let me know which|confirm (?:which|first|that)|first,? (?:tell|confirm|check with)|"
    r"do you mean|did you mean|do you want (?:me to|the|all|only|just)|should i\b|shall i\b|is it the)",
    re.IGNORECASE,
)
_LATE_ACT = re.compile(r"\bACT\s*\{")


def hedges(sentence: str) -> bool:
    """True when a sentence holds back: a question back to the person, a refusal, or acting only once they answer."""
    text = sentence.strip()
    return text.endswith("?") or bool(_HEDGE.search(text))


def promised_action(sentence: str) -> str:
    """The verb when a sentence promises to act ("I'll open Notepad", "I'll get that message to Priya"), else "".

    A sentence that promises and hedges at once ("I'll delete them once you tell me which") is not a promise.
    """
    if hedges(sentence):
        return ""
    for pattern in _PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group("verb").casefold()
    return ""


def late_act_at(text: str) -> int:
    """Where an ACT line starts after words in a reply (-1 when there is none). It is never spoken."""
    match = _LATE_ACT.search(text)
    return match.start() if match else -1


def act_start_held(text: str) -> int:
    """How many characters at the end of ``text`` could be the start of an ACT line, held back until the next piece
    of the stream shows whether they are (0 for ordinary words, so speech is never delayed by this)."""
    for size in range(min(len(text), 5), 0, -1):
        if text.endswith(("ACT {"[:size], "ACT{"[:size])):
            return size
    return 0


def declined(spoken: list[str], *, say: str = "") -> bool:
    """True when what the reply said (or the ACT line's own sentence) holds back: then nothing is handed over.

    Any question is not enough here ("Anything else?" after a promise still leaves the promise); a question about
    what was meant is ("Which ones do you mean?", "Do you want only the 2026 ones?").
    """
    return any(_HEDGE.search(sentence) for sentence in [*spoken, say] if sentence)
