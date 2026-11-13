"""A sentence the person repairs as they say it: "open, no, close Spotify", "set a timer for ten, no, fifteen minutes".

People fix what they are saying mid-sentence with a marker ("no", "no wait", "sorry", "I mean", "actually", "or
rather") and the repair. Only the repaired request is acted on, never both halves:

* the repair starts with a verb of its own: it is the whole new request ("open, no, close Spotify" -> "close
  Spotify");
* the repair is a number: it replaces the last number before the marker, and the rest of the sentence goes on from
  the repair ("remind me at four, no, five to call mum" -> "remind me at five to call mum");
* otherwise the repair replaces what the first part named after its verb ("open Notepad, no, Spotify" -> "open
  Spotify"; "delete the logs, no wait, the screenshots" -> "delete the screenshots");
* nothing after the marker ("open Spotify, no"): the request is taken back and nothing is acted on.

Anything that does not clearly read as a repair (no marker, the marker at the start, "no more than", a first part with
no verb) is left as said. A destructive repaired request still goes through the planner and its approval, which shows
what will be done.
"""

from __future__ import annotations

import re

from pilot.conversation.router import ACTION_VERBS

# The marker, between commas (as recognisers write a spoken pause), or "no"/"I mean" right before a verb.
_MARKER = r"(?:no,? wait|no,? sorry|no no|no|wait|sorry|i mean|actually|or rather|rather|uh|um|er)"
_SPLIT = re.compile(rf",\s*{_MARKER}\s*(?:,\s*|\s+|$)", re.IGNORECASE)
_BARE_SPLIT = re.compile(rf"\s+(?:no|i mean|sorry)\s+(?=(?:{'|'.join(map(re.escape, ACTION_VERBS))})\b)", re.IGNORECASE)
# "open Notepad no Spotify", written with no commas: a short object after "no" at the end, with a verb at the start.
_BARE_OBJECT = re.compile(r"\s+no\s+(?=(?:\S+\s*){1,3}$)", re.IGNORECASE)
_QUESTION = re.compile(r"^(?:what|when|where|who|which|how|why|is|are|do|does|can)\b", re.IGNORECASE)
_PREPOSITION = r"(?:to|for|in|from|at|on|with|about|into|by)"
_NOT_A_REPAIR = re.compile(r"\bno (?:more|longer|one|matter|need|idea|problem|way|thanks)\b", re.IGNORECASE)
_FILLERS = {"uh", "um", "er"}
_NUMBER = r"(?:\d+(?::\d\d)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty)"
_LAST_NUMBER = re.compile(rf"\b{_NUMBER}\b(?!.*\b{_NUMBER}\b)", re.IGNORECASE)
_STARTS_WITH_NUMBER = re.compile(rf"^{_NUMBER}\b", re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:the|a|an|my|this|that|these|those)\b", re.IGNORECASE)


def _verb(words: list[str]) -> str:
    return words[0].casefold() if words and words[0].casefold() in ACTION_VERBS else ""


def resolve(text: str) -> str | None:
    """The request as the person ended up meaning it, "" when they took it back, or None when the sentence holds no
    repair (it is acted on as said)."""
    said = " ".join(text.strip().split())
    if not said or _NOT_A_REPAIR.search(said):
        return None
    match = None
    for found in _SPLIT.finditer(said):
        match = found  # the last repair wins: "open, no, close, no, minimise Spotify"
    if match is None:
        match = _BARE_SPLIT.search(said)
    if match is None and said.split()[0].casefold() in ACTION_VERBS and len(said.split()) >= 4:
        match = _BARE_OBJECT.search(said)
    if match is None:
        return None
    first = said[: match.start()].strip(" ,")
    repair = said[match.end() :].strip(" ,.!?")
    marker = match.group(0).strip(" ,").casefold()
    first_words, repair_words = first.split(), repair.split()
    if first_words and repair_words and _QUESTION.match(first) and _QUESTION.match(repair):
        return repair  # "what time is it, no, what day is it"
    if not first_words or not _verb(first_words) and not _verb(repair_words):
        return None  # "No, stop" or a first part that asks for nothing: not a repair of a request
    if marker in _FILLERS and not _verb(repair_words):
        return None  # "uh" is a pause, not a repair, unless a new verb follows it
    if not repair_words:
        return ""  # "open Spotify, no": taken back
    if _verb(repair_words):
        return repair
    if _STARTS_WITH_NUMBER.match(repair) and _LAST_NUMBER.search(first):
        number = _LAST_NUMBER.search(first)
        assert number is not None
        return first[: number.start()] + repair
    verb = _verb(first_words)
    if not verb:
        return None
    starts = re.match(rf"^{_PREPOSITION}\b", repair, re.IGNORECASE)
    if starts:  # "email Ana about the invoice, no, about the report": from the last "about" on
        at = list(re.finditer(rf"\b{re.escape(starts.group(0))}\b", first, re.IGNORECASE))
        if at:
            return first[: at[-1].start()] + repair
    inner = list(re.finditer(rf"\s{_PREPOSITION}\s", first, re.IGNORECASE))
    if inner and len(repair_words) <= 3 and not _ARTICLE.match(repair):
        return first[: inner[-1].end()] + repair  # "search for flights to Lisbon, no, Porto": to Porto
    if len(first_words) > 1 and first_words[1].casefold() in {"up", "on", "off", "down"} and not _ARTICLE.match(repair):
        return f"{first_words[0]} {first_words[1]} {repair}"  # "turn off Wi-Fi, no, Bluetooth"
    return f"{first_words[0]} {repair}"
