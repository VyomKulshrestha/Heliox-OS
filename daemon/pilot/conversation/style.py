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
- Usually one sentence, at most two, at most 20 words in all. Answer first; add a reason only if asked.
- Say numbers, dates and file names the way a person says them aloud: "the backup from the twelfth of \
November", "sixteen gigabytes", never a file name or a date read out character by character.
- When Heliox Live spoke up on its own (marked in the conversation), it was pointing out something in what \
the person wrote or said. If they ask why, explain that evidence; Heliox did not make the mistake.
- If the person wants more, they will ask.
- Plain spoken words. No markdown, lists, headings, code, emoji, symbols or URLs. Never read out a file path; \
say the file or folder name instead.
- No stock openers or fillers ("Sure", "Great question", "Certainly", "Of course", "As an AI"). Don't repeat \
what the person just said back to them.
- Use what you are told about the person, the time and their screen when it helps; don't recite it. Use their \
name rarely, not in every reply.
- Write only the words Heliox says to the person. Never talk about them in the third person, never describe \
what you are about to do in the reply ("I'll ask a short question"), never think aloud.
- If you don't know, say so in a few words. If you misheard or the words don't make sense, say what you heard \
and ask what they meant. Never invent facts about their files, messages or calendar.

What you can and cannot do:
- In this reply you can only talk. Heliox's planner does things on the computer (opening apps, files, settings, \
reminders, searches, messages) after reviewing each plan for safety.
- Never say you did, opened, changed, sent or set something. Only the planner does that, and you are told when \
it has.
- You cannot save or remember anything for later, and you cannot change how Heliox behaves or its settings. \
Heliox does those in its own code before you are asked: what it keeps about the person is in what you are \
told about them, and a turn it handled shows as "(answered on this computer; done there: ...)" with what it \
did. When the person tells you something to remember, or asks Heliox to call them something or behave \
differently, and nothing you are told shows it was done, never say it was: say plainly that you couldn't \
save that, and that they can say "remember that ..." or "call me ...", or change it in Settings.
- If the person asks for something to be done on the computer, and you are sure what they want, reply with \
exactly one line and nothing else:
ACT {"verb": "<one verb>", "object": "<what it applies to, in their words>", "request": "<the full request, \
written out so it makes sense on its own>", "say": "<a short spoken sentence saying what you are about to do>"}
- When it is exactly one of these simple steps, add "action" in the planner's schema, for example \
"action": {"action_type": "open_application", "target": "notepad", "parameters": {"name": "notepad"}}. Simple \
steps are: open_application (an app by name, no arguments), open_url (a web address), window_focus \
(parameters: title), volume_set (level), volume_mute (mute), keyboard_press with a media key (playpause, \
nexttrack, prevtrack, volumeup, volumedown), battery_info, disk_usage, memory_usage, cpu_usage. Anything else: \
leave "action" out and the planner will work out the steps.
- If you are not sure what they want done, ask one short question instead. Never hand over a guess.
- Voice cannot approve anything. If an approval is waiting, say it is on the screen and the person presses \
Approve there. Denying or stopping by voice is fine.
- When the person asks what is happening or what you are waiting for, answer from what Heliox is doing (a \
running task, an approval on the screen) before anything else.
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


_UNITS = {
    "kb": "kilobytes",
    "mb": "megabytes",
    "gb": "gigabytes",
    "tb": "terabytes",
    "ghz": "gigahertz",
    "mhz": "megahertz",
    "ms": "milliseconds",
}
_GLUED_UNIT = re.compile(r"\b(\d+(?:\.\d+)?)\s?(kb|mb|gb|tb|ghz|mhz|ms)\b", re.IGNORECASE)
_PERCENT = re.compile(r"(\d)\s?%")
_DATED_FILE = re.compile(r"\b([A-Za-z][\w-]*?)[-_](\d{4})-(\d{2})-(\d{2})(?:\.[A-Za-z0-9]{1,5})?\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip


def _date_words(year: str, month: str, day: str) -> str | None:
    from pilot.live.spoken import date_words

    m, d = int(month), int(day)
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return date_words(d, _MONTHS[m - 1])


def speakable_numbers(text: str) -> str:
    """Units, percentages, dated file names and ISO dates in the words a voice says well."""

    def dated_file(match: re.Match[str]) -> str:
        words = _date_words(match.group(2), match.group(3), match.group(4))
        name = match.group(1).replace("-", " ").replace("_", " ")
        return f"the {name} from {words}" if words else match.group(0)

    def iso(match: re.Match[str]) -> str:
        return _date_words(*match.groups()) or match.group(0)

    out = _DATED_FILE.sub(dated_file, text)
    out = _ISO_DATE.sub(iso, out)
    out = _GLUED_UNIT.sub(lambda m: f"{m.group(1)} {_UNITS[m.group(2).casefold()]}", out)
    return say_numbers(out)


# How a voice says what a recogniser or a model writes. A speech engine reads "4 200" (a recogniser's thousands
# space) as "four, two hundred", "4200.00" with its zeros, and "3pm" letter by letter; a person says "four thousand
# two hundred", "three p m". Each shape is converted only where it cannot be anything else: a version ("v1.2.0"), an
# address ("192.168.1.1"), a plain year or count is left for the voice, which says those well.
_MONEY = re.compile(r"(?<![\w.])([$£€₹])\s?(\d{1,3}(?:[, \u00a0\u202f]\d{3})+|\d+)(?:\.(\d{2}))?(?![\d.]*\d)")
_CURRENCY = {"$": ("dollar", "cent"), "£": ("pound", "penny"), "€": ("euro", "cent"), "₹": ("rupee", "paisa")}
_MERIDIEM = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?\s?([ap])(?:\.\s?m\.|\s?m\b)", re.IGNORECASE)
_CLOCK = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_ORDINAL = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_GROUPED = re.compile(r"(?<![\w.,])(\d{1,3}(?:(?:,|[ \u00a0\u202f](?=\d{3}\b))\d{3})+)(\.\d+)?(?![\w,]|\.\d)")
_ZERO_CENTS = re.compile(r"(?<![\w.])(\d+)\.00(?![\w.])")
_PERCENT_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?%")


def say_numbers(text: str) -> str:
    """Amounts, times, ordinals and percentages as words: "4 200" -> "four thousand two hundred", "10:30" -> "ten
    thirty", "3pm" -> "three p m", "19th" -> "nineteenth", "50%" -> "fifty percent". Idempotent."""
    from pilot.live.spoken import cardinal_words, number_words, ordinal_words, time_words

    def plain(digits: str) -> int:
        return int(re.sub(r"[, \u00a0\u202f]", "", digits))

    def money(match: re.Match[str]) -> str:
        unit, small = _CURRENCY[match.group(1)]
        whole = plain(match.group(2))
        said = f"{cardinal_words(whole)} {unit}{'' if whole == 1 else 's'}"
        cents = int(match.group(3) or 0)
        if cents:
            small_word = small if cents == 1 else ("pence" if small == "penny" else small + "s")
            said += f" and {cardinal_words(cents)} {small_word}"
        return said

    def meridiem(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12:
            return match.group(0)
        return f"{time_words(hour, minute)} {match.group(3).casefold()} m"

    def clock(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour == 0:
            return f"{time_words(12, minute)} a m"
        if hour > 12:
            return f"{time_words(hour - 12, minute)} p m"
        return time_words(hour, minute)

    def grouped(match: re.Match[str]) -> str:
        return number_words(f"{plain(match.group(1))}{match.group(2) or ''}")

    out = _MONEY.sub(money, text)
    out = _MERIDIEM.sub(meridiem, out)
    out = _CLOCK.sub(clock, out)
    out = _ORDINAL.sub(lambda m: ordinal_words(int(m.group(1))) if 1 <= int(m.group(1)) <= 31 else m.group(0), out)
    out = _GROUPED.sub(grouped, out)
    out = _ZERO_CENTS.sub(lambda m: cardinal_words(int(m.group(1))), out)
    return _PERCENT_NUMBER.sub(lambda m: f"{number_words(m.group(1))} percent", out)


def clean_for_speech(text: str, *, first: bool = False) -> str:
    """Make one sentence of model text fit to be heard: no markdown, paths, links, symbols or stock openers."""
    out = speakable_numbers(text)
    out = _URL.sub("the link", out)
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
    """A sentence that claims an action the fast model cannot have taken, or narrates instead of speaking."""
    return bool(_CLAIMS_ACTION.search(sentence) or _NARRATION.search(sentence))


# The model thinking aloud or describing the person instead of speaking to them (seen in the conversation eval:
# "The person asked to put on music, which is ambiguous... so I'll ask one short question.").
_NARRATION = re.compile(
    r"\bthe (?:person|user) (?:asked|wants|said|is asking|seems|might|means|meant)\b|"
    r"\bi(?:'ll| will) ask (?:one|a|them)\b|\bso i(?:'ll| will) (?:ask|reply|answer|respond)\b|"
    r"\b(?:my|the) (?:reply|answer|response) (?:should|will)\b",
    re.IGNORECASE,
)


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
    action: dict | None = None  # one action in the planner's schema, when the model proposed one


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
    action = data.get("action") if isinstance(data.get("action"), dict) else None
    return ActRequest(
        action=action,
        verb=verb[:30],
        object=" ".join(str(data.get("object") or "").split())[:200],
        request=request[:500],
        say=say[:200],
    )


TYPED_SYSTEM_PROMPT = """\
You are Heliox, a companion that lives on the person's desktop computer. The person typed this in Heliox's chat \
window, which shows markdown.

How to write:
- Short and direct: answer first, a few sentences at most. Use a list or a code block only when it helps.
- No stock openers or fillers ("Sure", "Great question", "Certainly", "As an AI").
- Use what you are told about the person, the time and their screen when it helps; don't recite it. Use their \
name rarely, not in every reply.
- Write only the words Heliox says to the person. Never talk about them in the third person, never describe \
what you are about to do in the reply ("I'll ask a short question"), never think aloud.
- If you don't know, say so. If the words don't make sense, say what you read and ask what they meant. Never \
invent facts about their files, messages or calendar.
""" + SYSTEM_PROMPT[SYSTEM_PROMPT.index("What you can and cannot do:") :].replace(
    "If an approval is waiting, say it is on the screen and the person presses Approve there. Denying or "
    "stopping by voice is fine.",
    "If an approval is waiting, say it is on the screen and the person presses Approve there.",
)


_CHUNK_END = re.compile(r"(?:[.!?][\"')\]]*\s|\n)")


class TextChunker:
    """Typed replies: hands back text up to the last sentence end or line break, characters unchanged."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, fragment: str) -> list[str]:
        self._buffer += fragment
        ends = list(_CHUNK_END.finditer(self._buffer))
        if not ends:
            return []
        cut = ends[-1].end()
        chunk, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return [chunk]

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer, ""
        return [rest] if rest.strip() else []


def drop_action_claims(text: str) -> str:
    """Remove sentences that claim an action the fast model cannot have taken, keeping the rest's layout."""
    parts = re.split(r"(?<=[.!?])(\s+)", text)
    return "".join(part for part in parts if not claims_an_action(part))


# The model saying it will act, in words, without the ACT line that hands the request over.
_PROMISE = re.compile(
    r"\b(?:i'll|i will|let me|i'm going to|i am going to|i can do that now and)\s+(?:go ahead and\s+|just\s+)?"
    r"(?P<verb>open|close|delete|move|copy|send|launch|start|set|create|turn|mute|unmute|play|pause|search|find|"
    r"install|uninstall|remove|rename|run|schedule|book|empty|switch|lock|restart|shut|save|download)\b",
    re.IGNORECASE,
)


def promised_action(sentence: str) -> str:
    """The verb when a sentence promises to act ("I'll open Notepad"), else ""."""
    match = _PROMISE.search(sentence)
    return match.group("verb").casefold() if match else ""
