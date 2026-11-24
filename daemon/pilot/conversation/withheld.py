"""A secret said aloud stays on this computer: no model ever gets it, and nothing keeps it.

"Oh, and my bank PIN is 4471." Heliox never says it back, and until this rule the number still went to the talk
model as the person's own turn in every later request. The rule is the one the never-kept rule uses for the profile
(``pilot.personal.never_kept``): a sentence that names a secret (a PIN, a password or passcode, a key or token, a
security, door, alarm, verification or recovery code, a login or username, a card or account number, a CVV, a sort
code, an IBAN or routing number, a social security, passport or licence number) together with something that can be
its value, or anything that looks like a key, token or card number on its own
(``pilot.workflows.clipboard_history.looks_secret``, the Luhn check for card numbers).

What a model gets in its place: the words with each value replaced by ``[left out: a PIN]``, so the conversation
still reads in order ("oh, and my bank PIN is [left out: a PIN]"). That goes to every model alike, one on this
computer too: Heliox does nothing with a secret said aloud, so no model needs it, and a talk model on this computer
can still hand a request to a planner in the cloud. When the value cannot be told apart from the rest of the
sentence, the whole sentence is left out.

``ConversationMemory`` keeps only the left-out form, and also takes the value out of anything later added to it
(Heliox repeating it, a code answer that quotes it). The values themselves are held in that object's memory only,
to recognise them, and are gone with "new conversation".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# What each secret name is called in the marker. Checked in order; the first that matches the sentence wins.
_KINDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpin\b", re.I), "a PIN"),
    (re.compile(r"\b(?:password|passcode|passphrase)s?\b", re.I), "a password"),
    (re.compile(r"\b(?:card number|cvv|cvc|security code)s?\b", re.I), "a card number"),
    (re.compile(r"\b(?:account number|sort code|iban|routing number)s?\b", re.I), "an account number"),
    (re.compile(r"\b(?:social security|ssn|passport number|licen[cs]e number)s?\b", re.I), "an ID number"),
    (re.compile(r"\b(?:login|username)s?\b", re.I), "a login"),
    (re.compile(r"\b(?:keys?|tokens?)\b", re.I), "a key"),
)
DEFAULT_KIND = "a code"
_DIGIT_RUN = re.compile(r"(?<![\w])\d[\d \-.]*\d(?![\w])|(?<![\w])\d{3,}(?![\w])")
_NUMBER_WORD = (
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen"
    r"|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|double"
    r"|triple)"
)
_NUMBER_WORDS = re.compile(rf"\b{_NUMBER_WORD}(?:[\s-]+{_NUMBER_WORD}){{2,}}\b", re.I)
# "my wifi password is swordfish": the one word after the secret's name and "is", when it ends the clause and is not
# an ordinary word ("my password is wrong", "my PIN is on a card").
_AFTER_IS = re.compile(r"\b(?:is|was|=|:)\s+(?P<value>[^\s,;.!?]+)(?=\s*(?:$|[,;.!?]|\s+(?:and|but|so|then)\b))", re.I)
_ORDINARY = frozenset(
    [
        "wrong",
        "right",
        "correct",
        "changed",
        "expired",
        "weak",
        "strong",
        "long",
        "short",
        "too",
        "not",
        "the",
        "a",
        "an",
        "my",
        "your",
        "in",
        "on",
        "at",
        "still",
        "now",
        "new",
        "old",
        "same",
        "different",
        "easy",
        "hard",
        "fine",
        "ok",
        "okay",
        "safe",
        "secret",
        "written",
        "saved",
        "stored",
        "somewhere",
        "here",
        "there",
        "what",
        "this",
        "that",
        "it",
        "gone",
        "lost",
        "forgotten",
        "locked",
        "blocked",
        "working",
        "broken",
        "invalid",
        "incorrect",
        "required",
        "needed",
        "set",
        "empty",
        "blank",
        "very",
        "really",
        "so",
        "just",
        "also",
        "mine",
        "yours",
        "his",
        "hers",
        "theirs",
        "ours",
        "missing",
        "useless",
        "fake",
        "public",
        "private",
        "simple",
        "complicated",
        "reset",
    ]
)
MARK = "[left out: {kind}]"
MARK_START = "[left out: "


@dataclass(frozen=True, slots=True)
class Withheld:
    kind: str  # "a PIN", "a password", ...
    text: str  # the words as a model may see them
    values: tuple[str, ...]  # what was taken out (kept in memory only, to recognise it again)


def _names_secret(text: str) -> bool:
    try:
        from pilot.personal.never_kept import _SECRET
    except Exception:  # pragma: no cover - the module is pure
        return False
    return bool(_SECRET.search(text))


def _kind(text: str) -> str:
    return next((kind for pattern, kind in _KINDS if pattern.search(text)), DEFAULT_KIND)


def _token_secret(token: str, named: bool) -> bool:
    """A key or token by its shape ("sk-ant-...", "ghp_..."). The looser test of mixed letters, digits and signs only
    when the sentence names a secret: on its own it takes file names ("q3-summary.docx") for keys."""
    from pilot.workflows.clipboard_history import _SECRET_PATTERNS, looks_secret

    if any(pattern.search(token) for pattern in _SECRET_PATTERNS):
        return True
    return named and looks_secret(token)


def _card_number(run: str) -> bool:
    from pilot.workflows.clipboard_history import _luhn

    digits = re.sub(r"\D", "", run)
    return 13 <= len(digits) <= 19 and _luhn(digits)


def _values(text: str, named: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        token = match.group(0).strip(".,;:!?\"'()[]")
        if len(token) >= 8 and _token_secret(token, named):
            spans.append((match.start(), match.end()))
    for match in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if (named and len(digits) >= 3) or _card_number(match.group(0)):
            spans.append(match.span())
    if named:
        spans += [match.span() for match in _NUMBER_WORDS.finditer(text)]
        for match in _AFTER_IS.finditer(text):
            value = match.group("value")
            if value.casefold() not in _ORDINARY and not value.startswith(MARK_START):
                spans.append(match.span("value"))
    return spans


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def withhold(text: str) -> Withheld | None:
    """The words with every secret taken out, or None when they hold none. Pure, and well under a millisecond."""
    if not text or not text.strip():
        return None
    named = _names_secret(text)
    spans = _merge(_values(text, named))
    if not spans:
        return None
    kind = _kind(text) if named else ("a card number" if any(_card_number(text[a:b]) for a, b in spans) else "a key")
    mark = MARK.format(kind=kind)
    out, last, values = [], 0, []
    for start, end in spans:
        out.append(text[last:start])
        out.append(mark)
        values.append(text[start:end].strip())
        last = end
    out.append(text[last:])
    said = "".join(out)
    if _values(said.replace(mark, ""), named):  # something value-like left: the whole sentence goes
        said = f"[left out: a sentence holding {kind}]"
    return Withheld(kind=kind, text=said, values=tuple(value for value in values if value))


def without_values(text: str, values: set[str] | frozenset[str]) -> str:
    """``text`` with each value already taken out of this conversation replaced by the marker."""
    for value in sorted(values, key=len, reverse=True):
        if value and value in text:
            text = text.replace(value, MARK.format(kind="something private"))
    return text


def holds_secret(text: str) -> bool:
    return withhold(text) is not None


__all__ = ["MARK_START", "Withheld", "holds_secret", "withhold", "without_values"]
