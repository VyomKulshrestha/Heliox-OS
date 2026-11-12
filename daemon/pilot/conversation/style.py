"""How Heliox sounds: the fast model's instructions, the clean-up of its text for the ear, and sentence cutting.

Everything here is pure text processing, so it is cheap enough to run on every streamed fragment and is tested
without a model or a speaker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SYSTEM_PROMPT = """\
You are Heliox, a voice companion that lives on the person's desktop computer. Everything you write is spoken \
aloud by a text-to-speech voice the moment you write it, so write the way a sensible person talks.

How to talk:
- Usually one sentence, at most two, each under twenty words. Answer first; add a reason only if asked.
- If the person wants more, they will ask.
- Plain spoken words. No markdown, lists, headings, code, emoji, symbols or URLs. Never read out a file path; \
say the file or folder name instead.
- No stock openers or fillers ("Sure", "Great question", "Certainly", "Of course", "As an AI"). Don't repeat \
what the person just said back to them.
- Use what you are told about the person, the time and their screen when it helps; don't recite it.
- If you don't know, say so in a few words. If you misheard or the words don't make sense, say what you heard \
and ask what they meant. Never invent facts about their files, messages or calendar.

What you can and cannot do:
- In this reply you can only talk. Heliox's planner does things on the computer (opening apps, files, settings, \
reminders, searches, messages) after reviewing each plan for safety.
- Never say you did, opened, changed, sent or set something. Only the planner does that, and you are told when \
it has.
- If the person asks for something to be done on the computer, and you are sure what they want, reply with \
exactly one line and nothing else:
ACT {"verb": "<one verb>", "object": "<what it applies to, in their words>", "request": "<the full request, \
written out so it makes sense on its own>", "say": "<a short spoken sentence saying what you are about to do>"}
- If you are not sure what they want done, ask one short question instead. Never hand over a guess.
- Voice cannot approve anything. If an approval is waiting, say it is on the screen and the person presses \
Approve there. Denying or stopping by voice is fine.
"""

ACT_PREFIX = "ACT"

# Openers that add nothing when heard. Removed only at the very start of a reply.
_STOCK_OPENERS = re.compile(
    r"^\s*(?:(?:sure|certainly|of course|absolutely|great question|good question|okay so|so|well|alright|"
    r"as an ai(?: language model)?|i'd be happy to help|happy to help|no problem)(?:[,.!:]?\s+|[,.!:]?\s*$))+",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"(\*\*|__|\*|`+|^#{1,6}\s*|^\s*>\s*)", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", re.MULTILINE)
_URL = re.compile(r"\b(?:https?://|www\.)\S+?(?=[.,;:!?)\]]*(?:\s|$))", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s,;]+\\)*([^\\\s,;]*)")
_POSIX_PATH = re.compile(r"(?<![\w.])(?:~|/(?:home|users|usr|etc|var|tmp|opt|mnt|c))(?:/[^\s/,;]+)+", re.IGNORECASE)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿️]")
_SYMBOLS = {" & ": " and ", " -> ": " to ", " => ": " to ", "→": " to ", " + ": " plus ", " = ": " is "}
# The fast model can only talk. A sentence claiming it did something is not true and is not spoken.
_CLAIMS_ACTION = re.compile(
    r"\b(?:i(?:'ve| have)|i just|i've just|done[,.!]? i)\s+(?:just\s+)?(?:opened|closed|deleted|removed|sent|"
    r"created|moved|copied|renamed|saved|set|started|launched|installed|turned|scheduled|added|emailed|"
    r"booked|changed|updated|fixed)\b",
    re.IGNORECASE,
)


def _path_name(match: re.Match[str]) -> str:
    text = match.group(0).rstrip("\\/")
    name = re.split(r"[\\/]", text)[-1]
    return name or "that folder"


def clean_for_speech(text: str, *, first: bool = False) -> str:
    """Make one sentence of model text fit to be heard: no markdown, paths, links, symbols or stock openers."""
    out = _URL.sub("the link", text)
    out = _WINDOWS_PATH.sub(_path_name, out)
    out = _POSIX_PATH.sub(_path_name, out)
    out = _BULLET.sub("", out)
    out = _MARKDOWN.sub("", out)
    out = _EMOJI.sub("", out)
    for symbol, words in _SYMBOLS.items():
        out = out.replace(symbol, words)
    if first:
        out = _STOCK_OPENERS.sub("", out)
        if out[:1].islower():
            out = out[:1].upper() + out[1:]
    return " ".join(out.split())


def claims_an_action(sentence: str) -> bool:
    return bool(_CLAIMS_ACTION.search(sentence))


# Words after which a full stop does not end a sentence.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "eg", "ie", "approx", "no", "fig", "inc", "ltd"}
)
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
FIRST_CLAUSE_MIN = 70  # a long first sentence is cut at a comma after this many characters, so sound starts sooner
MAX_SENTENCE = 240


class SentenceSplitter:
    """Collects streamed text and gives back whole sentences as soon as each one ends."""

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = 0

    def feed(self, fragment: str) -> list[str]:
        self._buffer += fragment
        out: list[str] = []
        while True:
            sentence = self._take()
            if sentence is None:
                return out
            out.append(sentence)

    def flush(self) -> list[str]:
        rest = " ".join(self._buffer.split())
        self._buffer = ""
        return [rest] if rest else []

    def _take(self) -> str | None:
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            end = match.end()
            before = self._buffer[start : match.start()].rsplit(None, 1)
            last_word = before[-1].casefold().rstrip(".") if before else ""
            if match.group(0).startswith(".") and (
                last_word in _ABBREVIATIONS or (last_word[-1:].isdigit() and self._buffer[end:].lstrip()[:1].isdigit())
            ):
                continue
            return self._cut(end)
        limit = FIRST_CLAUSE_MIN if self._emitted == 0 else MAX_SENTENCE
        if len(self._buffer) > limit:
            comma = max(self._buffer.rfind(", ", 0, MAX_SENTENCE), self._buffer.rfind("; ", 0, MAX_SENTENCE))
            if comma >= limit // 2:
                return self._cut(comma + 1)
            if len(self._buffer) > MAX_SENTENCE:
                space = self._buffer.rfind(" ", 0, MAX_SENTENCE)
                return self._cut(space if space > 0 else MAX_SENTENCE)
        return None

    def _cut(self, end: int) -> str | None:
        sentence = " ".join(self._buffer[:end].split())
        self._buffer = self._buffer[end:]
        if not sentence:
            return None
        self._emitted += 1
        return sentence


@dataclass(frozen=True, slots=True)
class ActRequest:
    """The fast model's "this needs an action", in a form a planner can take without the conversation."""

    verb: str
    object: str
    request: str
    say: str


def looks_like_act(prefix: str) -> bool | None:
    """True once the reply is clearly an ACT line, False once it clearly is not, None while too short to tell."""
    stripped = prefix.lstrip()
    if len(stripped) < len(ACT_PREFIX) + 1:
        return None if ACT_PREFIX.startswith(stripped) or not stripped else False
    return stripped.startswith(ACT_PREFIX) and stripped[len(ACT_PREFIX)] in " {"


def parse_act(text: str) -> ActRequest | None:
    """Read an ``ACT {...}`` line. Anything malformed is None, and the caller treats the reply as unclear."""
    stripped = text.strip()
    if not stripped.startswith(ACT_PREFIX):
        return None
    body = stripped[len(ACT_PREFIX) :].strip()
    end = body.rfind("}")
    if not body.startswith("{") or end < 0:
        return None
    try:
        data = json.loads(body[: end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    verb = str(data.get("verb") or "").strip().casefold()
    request = " ".join(str(data.get("request") or "").split())
    if not verb or not request:
        return None
    say = clean_for_speech(str(data.get("say") or ""), first=True)
    if claims_an_action(say):
        say = ""
    return ActRequest(
        verb=verb[:30],
        object=" ".join(str(data.get("object") or "").split())[:200],
        request=request[:500],
        say=say[:200],
    )
