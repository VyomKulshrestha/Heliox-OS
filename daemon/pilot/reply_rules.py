"""The rules for what Heliox shows and says about a finished task, kept in one place.

A reply has two forms. The shown one goes to the window; it may name a file by its path, but it is never a handler's
raw JSON and never a path with its backslashes doubled by a ``repr``. The spoken one goes to the voice: never a hash,
an id, a full path, JSON, brackets or backslashes; numbers in a form a voice says well; at most ``SPOKEN_LIMIT``
characters, cut at the end of a sentence and never inside a word.

``for_ear`` and ``for_screen`` apply the rules; ``ear_faults`` and ``screen_faults`` name what breaks them, for the
contract test (tests/test_reply_contract.py) and for the last check before a reply leaves the daemon.
"""

from __future__ import annotations

import json
import re

SPOKEN_LIMIT = 200  # characters: about two short sentences, ten seconds of speech
SHOWN_LIMIT = 1200  # characters of a read's answer kept in the shown reply (the full output stays in the details)

# A hash or an id: hex of 7 characters or more holding a digit and a letter (git's short hash, a trigger id, a
# digest), a UUID, or 20 or more hex digits of any kind.
_HEX_ID = re.compile(
    r"(?<![\w-])(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}|[0-9a-f]{20,})(?![\w-])",
    re.IGNORECASE,
)
_ALGORITHM = re.compile(r"\b(?:sha-?(?:1|224|256|384|512)|md5|blake2[bs]?)\b[:\s]*", re.IGNORECASE)
_URL = re.compile(r"\b(?:https?|ftp|wss?)://(?:[^\s@/]+@)?(?P<host>[^\s/:?#\"'<>]+)(?::\d+)?[^\s\"'<>]*", re.I)
# A path: a drive letter, a UNC share, a home folder or an absolute path of two parts or more. Quotes around it are
# kept apart. A name with spaces inside a path cannot be told from the words after it, so it ends at the space.
_PATH = re.compile(
    r"(?:(?<![\w])[A-Za-z]:[\\/]|\\\\(?=[\w.$-])|(?<![\w.])~[\\/]|(?<![\w.:/])/(?=[^\s/]+/))[^\s\"'<>|;,()\[\]{}]*"
)
_SLASHED = re.compile(r"[^\s\"'<>|;,()\[\]{}]*[\\/][^\s\"'<>|;,()\[\]{}]*")
_FILE_NAME = re.compile(r"\.[A-Za-z0-9]{1,6}$")
_UNDERSCORED = re.compile(r"\S*\w_\w\S*")
_DATA = re.compile(r"[\[{]\s*[\[{]?\s*['\"][\w ]+['\"]\s*:")
_SHORT_WORDS = re.compile(r"^[A-Za-z]{1,6}$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SYMBOLS = (
    (re.compile(r"\s*(?:->|=>|→)\s*"), " to "),
    (re.compile(r"\s+&\s+"), " and "),
    (re.compile(r"\s*\|\s*"), ", "),
    (re.compile(r"[{}\[\]<>\\`*#]"), " "),
    (re.compile(r"\s*\(\s*\)\s*"), " "),
)
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_TIDY = (
    (re.compile(r"\s+([,.;:!?])(?=\s|$)"), r"\1"),  # a space before a mark that ends a clause, not ".price-table"
    (re.compile(r"([,;:])(?:\s*[,;:])+"), r"\1"),
    (re.compile(r"[,;:]\s*([.!?])"), r"\1"),
    (re.compile(r"([.!?])(?:\s*[.,;:])+"), r"\1"),
    (re.compile(r"^[\s,.;:!?-]+"), ""),
    (re.compile(r"\s{2,}"), " "),
)


def looks_like_json(text: str) -> bool:
    """A handler's structured output: a JSON object or list, or a printed Python dict or list of dicts, alone or after
    a short label ("Active monitors: [{'task_id': ..."). Data quoted inside a longer sentence (an error that prints a
    dict) is not counted: it is shown as the system wrote it."""
    body = (text or "").strip()
    found = re.search(r"[\[{]", body)
    if not found or found.start() > 40:
        return False
    if found.start() == 0:
        try:
            json.loads(body)
            return True
        except ValueError:
            pass
    return bool(_DATA.match(body, found.start()))


def parsed_json(text: str) -> object | None:
    """The handler's output as JSON data, or None."""
    body = (text or "").strip()
    if not body or body[0] not in "{[":
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def undoubled(text: str) -> str:
    """A path printed by ``repr`` (or JSON inside a repr) has its backslashes doubled, or doubled again: put back."""
    out = str(text or "")
    while "\\\\" in out:
        out = out.replace("\\\\", "\\")
    return out


def name_of(path: str) -> str:
    """The last part of a path ("note.txt" of C:\\work\\notes\\note.txt); the path itself when it has one part."""
    text = undoubled(str(path or "")).strip().strip("'\"").rstrip("\\/")
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z]:", text):
        return f"drive {text[0].upper()}"
    return re.split(r"[\\/]", text)[-1] or text


def folder_of(path: str) -> str:
    """The name of the folder a path is in, or ""."""
    text = undoubled(str(path or "")).strip().strip("'\"").rstrip("\\/")
    parts = [part for part in re.split(r"[\\/]", text) if part]
    if len(parts) < 2:
        return ""
    folder = parts[-2]
    return f"drive {folder[0].upper()}" if re.fullmatch(r"[A-Za-z]:", folder) else folder


def host_of(url: str) -> str:
    found = _URL.search(str(url or ""))
    host = found.group("host") if found else str(url or "").strip()
    return host.removeprefix("www.").rstrip(".")


def plural(count: int, noun: str, many: str = "") -> str:
    return f"{count:,} {noun if count == 1 else (many or noun + 's')}"


def size_words(size: int | float) -> str:
    """40 bytes, 1.5 kilobytes, 300 megabytes: what a voice says, and what reads well on screen."""
    value = float(size)
    for unit in ("bytes", "kilobytes", "megabytes", "gigabytes", "terabytes"):
        if value < 1024 or unit == "terabytes":
            if unit == "bytes":
                return plural(int(value), "byte")
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
        value /= 1024
    return f"{size} bytes"  # pragma: no cover


def words_cut(text: str, count: int) -> str:
    """The first ``count`` words of a text, never part of a word."""
    words = str(text or "").split()
    if len(words) <= count:
        return " ".join(words)
    kept = " ".join(words[:count]).rstrip(",;:-")
    # Never cut right after a word that needs another ("...afternoon because the..." is said "because the.").
    while True:
        cut = _DANGLING_END.sub("", kept).rstrip(",;:-")
        if cut == kept or not cut:
            break
        kept = cut
    return kept + "..."


def _path_name(found: re.Match[str]) -> str:
    whole = found.group(0)
    trail = re.search(r"[.:]+$", whole)
    path = whole[: trail.start()] if trail else whole
    return name_of(path) + (trail.group(0) if trail else "")


def _slashed(found: re.Match[str]) -> str:
    token = found.group(0)
    parts = [part for part in re.split(r"[\\/]+", token) if part]
    if not parts:
        return " "
    if len(parts) >= 2 and all(_SHORT_WORDS.match(part) for part in parts) and "." not in token:
        return " ".join(parts) if "or" in parts else " or ".join(parts)  # "and/or", "on/off"
    if all(part.isdigit() for part in parts):
        return " of ".join(parts) if len(parts) == 2 else " ".join(parts)
    return parts[-1]


def _underscores(found: re.Match[str]) -> str:
    """file_modified -> "file modified"; a file name keeps its underscores (note_backup.txt)."""
    token = found.group(0)
    return token if _FILE_NAME.search(token.rstrip(".,;:!?'\"")) else token.replace("_", " ")


def _paren(found: re.Match[str]) -> str:
    inside = found.group(1).strip(" ,;")
    return f", {inside}," if inside else ""


_DANGLING_END = re.compile(
    r"\s+(?:the|a|an|its|your|my|their|our|and|but|or|than|because|that|which|who|when|if|to|of|in|on|at|by|for"
    r"|with|from|as|is|are|was|were)$",
    re.IGNORECASE,
)


def _sentences_within(text: str, limit: int) -> str:
    """Whole sentences up to ``limit`` characters; a first sentence longer than that is cut at a word."""
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in _SENTENCE_END.split(text):
        joined = f"{kept} {sentence}".strip()
        if len(joined) > limit:
            break
        kept = joined
    if kept:
        return kept
    head = text[: limit + 1]
    head = head[: head.rfind(" ")] if " " in head else head[:limit]
    head = head.rstrip(" ,;:-'\"")
    # Cut at a word, the sentence must not end on one that needs another after it ("...because the."): those go too.
    while True:
        cut = _DANGLING_END.sub("", head).rstrip(" ,;:-'\"")
        if cut == head or not cut:
            break
        head = cut
    return head + "."


def for_ear(text: str, *, limit: int = SPOKEN_LIMIT) -> str:
    """The reply as it may be said: no raw data, a file by its name, a web address by its host, no hash or id, no
    brackets or backslashes, numbers a voice says well, at most ``limit`` characters ending at a sentence's end.
    "" for raw JSON: the caller says its own sentence instead."""
    from pilot.conversation.style import speakable_numbers

    out = undoubled(str(text or "")).strip()
    if not out or looks_like_json(out):
        return ""
    lines = [line.strip(" -*") for line in out.splitlines() if line.strip(" -*")]
    out = " ".join(line if line[-1:] in ".!?:,;" else line + "." for line in lines)
    out = _URL.sub(lambda m: m.group("host").removeprefix("www."), out)
    out = _ALGORITHM.sub("", _HEX_ID.sub("", out)) if _HEX_ID.search(out) else out
    out = _PATH.sub(_path_name, out)
    out = _SLASHED.sub(_slashed, out)
    out = _PAREN.sub(_paren, out)
    for pattern, words in _SYMBOLS:
        out = pattern.sub(words, out)
    out = _UNDERSCORED.sub(_underscores, out)
    out = speakable_numbers(out.replace("@", " at "))
    for pattern, replacement in _TIDY:
        out = pattern.sub(replacement, out)
    out = out.strip(" ,;:-")
    if not out:
        return ""
    if out[-1] not in ".!?":
        out += "."
    out = out[:1].upper() + out[1:]
    return _sentences_within(out, limit)


def for_screen(text: str, *, limit: int = SHOWN_LIMIT) -> str:
    """The reply as it may be shown: backslashes as written, whitespace as it was, at most ``limit`` characters."""
    out = undoubled(str(text or "")).strip()
    if len(out) > limit:
        cut = out[: limit + 1]
        out = (cut[: cut.rfind(" ")] if " " in cut[limit // 2 :] else cut[:limit]).rstrip() + "..."
    return out


# The marks a filter leaves when it takes words out of a sentence and keeps the rest: "and it's now, so" (a time
# taken out), a space before a comma, a doubled space, a sentence that ends on a word that needs another after it.
_HOLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a time missing after now", re.compile(r"\b(?:it'?s|is|was) now\s*[,.;]", re.IGNORECASE)),
    # Not an ellipsis (" ...work"), and not a name or a selector that starts with a dot (".env", ".price-table",
    # "main .title"): a space left where a word was taken out before a comma or a full stop.
    ("a space before a mark", re.compile(r"\s[,;:!?]|\s\.(?![.\w])")),
    # Two spaces inside a sentence (two after a full stop are a typing habit, and indented lines are never spoken).
    ("a doubled space", re.compile(r"[^\s.!?:] {2,}\S")),
    (
        "a dangling word",
        # A word that needs another after it ("the", "and", "your"), right before the end. Prepositions, "is",
        # "so" and the like end good sentences ("a folder you can write to.", "Blue shirt it is.", "I think so."),
        # so they are not here.
        re.compile(
            r"(?<![\w'-])(?:the|an|its|your|my|their|our|and|but|or|than)\s*[.!?]\s*$",
            re.IGNORECASE,
        ),
    ),
)


def hole(text: str) -> str:
    """What marks ``text`` as a sentence with words taken out of it ("a time missing after now"), or "". A sentence
    that has one is never said: the speech path drops it (pilot.conversation.layer)."""
    spoken = str(text or "")
    return next((name for name, pattern in _HOLES if pattern.search(spoken)), "")


def ear_faults(text: str) -> list[str]:
    """What a spoken reply breaks, or []: the contract test fails on any of these."""
    faults = []
    spoken = str(text or "")
    if not spoken.strip():
        faults.append("empty")
    if _HEX_ID.search(spoken):
        faults.append("hash or id")
    if "\\" in spoken or "/" in spoken:
        faults.append("path separator")
    if any(mark in spoken for mark in "{}[]"):
        faults.append("bracket or brace")
    if looks_like_json(spoken):
        faults.append("json")
    if len(spoken) > SPOKEN_LIMIT:
        faults.append("too long")
    if spoken.strip() and spoken.strip()[-1] not in ".!?":
        faults.append("does not end a sentence")
    if hole(spoken):
        faults.append(f"hole: {hole(spoken)}")
    return faults


def screen_faults(text: str) -> list[str]:
    faults = []
    shown = str(text or "")
    if not shown.strip():
        faults.append("empty")
    if "\\\\" in shown:
        faults.append("doubled backslashes")
    if looks_like_json(shown):
        faults.append("json")
    return faults
