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
# Up to three words ("hunter two", "blue horse seven") when they end the clause.
_AFTER_IS = re.compile(
    r"\b(?:is|was|=|:)\s+(?P<value>[^\s,;.!?]+(?:\s+[^\s,;.!?]+){0,2}?)"
    r"(?=\s*(?:$|[,;.!?]|\s+(?:and|but|so|then|don'?t|do not)\b))",
    re.IGNORECASE,
)
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


# A web address's host and port ("http://127.0.0.1:8797/hook"), with no user name or password before an "@": its
# digits are an address, never a secret's value. In the live run of 26 November a typed request naming a bearer token
# had "127.0.0.1" and "8797" taken out as keys too.
_URL_HOST = re.compile(r"\b[a-z][a-z0-9+.-]*://(?:[^\s/@?#]*@)?(?P<host>[^\s/@?#]+)(?=[/\s?#]|$)", re.IGNORECASE)
# The user name and password a web address carries before its host: "https://sam:hunter2@host:8080/",
# "ftp://:4471@host", "https://ghp_...:x-oauth-basic@github.com/...". Only before the first "/", "?" or "#" after
# "://", so an "@" later in the path or query ("/users/@sam", "?to=sam@example.com") is never taken for one.
_URL_USERINFO = re.compile(
    r"\b(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^\s:/@?#]*)(?::(?P<password>[^\s/@?#]*))?@(?=[\w\[])",
    re.IGNORECASE,
)
# Words that stand where a password or user name goes and are not secrets: GitHub's old token form puts the token
# as the user name and "x-oauth-basic" as the password; its newer form puts "x-access-token" as the user name.
_PLACEHOLDERS = frozenset({"", "x-oauth-basic", "x-access-token", "oauth2", "token", "git"})
# A secret already taken out of a web address's user name or password ("https://sam:[left out: a password]@host").
_LEFT_OUT_USERINFO = re.compile(r"://[^\s/?#@]*\[left out: [^\]]+\][^\s/?#@]*@")


def _url_credentials(text: str) -> list[tuple[int, int]]:
    """The spans of the secrets a web address carries: its password (all of it: digits only, percent-encoded or not),
    or its user name when that is a token and the password is empty or a placeholder."""
    spans: list[tuple[int, int]] = []
    for match in _URL_USERINFO.finditer(text):
        user, password = match.group("user") or "", match.group("password")
        if password is not None and password.casefold() not in _PLACEHOLDERS:
            spans.append(match.span("password"))
        elif user.casefold() not in _PLACEHOLDERS and (len(user) >= 20 or _token_secret(user, True)):
            spans.append(match.span("user"))  # the token is the user name
    return spans


def holds_url_credential(text: str) -> bool:
    """Words with a password or token inside a web address, still there or already taken out."""
    return bool(_url_credentials(text or "") or _LEFT_OUT_USERINFO.search(text or ""))


def _in_address(text: str, span: tuple[int, int]) -> bool:
    return any(match.start("host") <= span[0] and span[1] <= match.end("host") for match in _URL_HOST.finditer(text))


def _values(text: str, named: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = _url_credentials(text)
    for match in re.finditer(r"\S+", text):
        token = match.group(0).strip(".,;:!?\"'()[]")
        if _URL_USERINFO.search(token):
            # A web address with a user name and password: those are taken out above, and the rest (host, port,
            # path, the user name) stays readable unless it is itself shaped like a key.
            token = _URL_USERINFO.sub(lambda found: found.group("scheme"), token)
        if len(token) >= 8 and _token_secret(token, named):
            spans.append((match.start(), match.end()))
    for match in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if ((named and len(digits) >= 3) or _card_number(match.group(0))) and not _in_address(text, match.span()):
            spans.append(match.span())
    if named:
        spans += [match.span() for match in _NUMBER_WORDS.finditer(text)]
        for match in _AFTER_IS.finditer(text):
            value = match.group("value")
            if value.split()[0].casefold() not in _ORDINARY and not value.startswith(MARK_START):
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
    if named:
        kind = _kind(text)
    elif any(_card_number(text[a:b]) for a, b in spans):
        kind = "a card number"
    else:
        kind = "a password" if _url_credentials(text) else "a key"
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


def holds_left_out(text: str) -> bool:
    """Words a secret was already taken out of (``left_out`` ran on them where they were first written)."""
    return MARK_START in (text or "")


# Values taken out at the first point the words existed, held in this process's memory for RECENT_SECONDS so the same
# number said again on its own ("was it 4471?") is taken out too. Never written anywhere.
RECENT_SECONDS = 600.0
_recent: dict[str, float] = {}


def left_out(text: str, *, now: float | None = None) -> str:
    """The words as everything after the recogniser (or a typed box, or a phone) may see them: each secret replaced by
    "[left out: a PIN]", and any value taken out in the last RECENT_SECONDS too. Called once, where the words first
    exist, so logs, notices, stores, Live and every model read only this form."""
    import time

    if not text:
        return text
    moment = time.monotonic() if now is None else now
    for value, at in list(_recent.items()):
        if moment - at > RECENT_SECONDS:
            _recent.pop(value, None)
    found = withhold(text)
    if found is not None:
        for value in found.values:
            _recent[value] = moment
        text = found.text
    return without_values(text, set(_recent)) if _recent else text


def forget_recent() -> None:
    """A new conversation: values taken out earlier are no longer looked for."""
    _recent.clear()


# What is said when a request held a secret and nothing was done (pilot.conversation.layer, the server's gates,
# the autonomous engine).
SECRET_NOT_USED = (
    "I don't use a PIN, password or code said aloud, so I left it out. Type it in yourself where it's needed."
)
# The services whose secret has its own place in Settings (pilot.system.connections.WHERE), by the words that name
# them in a request. Checked in order.
_SERVICE_WORDS = (
    ("slack", re.compile(r"\bslack\b", re.IGNORECASE)),
    ("discord", re.compile(r"\bdiscord\b", re.IGNORECASE)),
    ("github", re.compile(r"\bgit ?hub\b|\bgh_|\bpull request", re.IGNORECASE)),
    ("mail", re.compile(r"\b(?:e-?mail|mail|smtp|imap|inbox|gmail|outlook)\b", re.IGNORECASE)),
)
_LEFT_OUT_KIND = re.compile(r"\[left out: (?:a sentence holding )?(?:an? )?(?P<kind>[^\]]+)\]")


def secret_not_used(words: str, *, typed: bool, sent: bool = False) -> str:
    """What Heliox says when a request held a secret and nothing was done (the secret never goes to a model, and a
    request whose point was the secret cannot be done without it). Worded for how it arrived: typed, said aloud, or
    ``sent`` from somewhere else (a phone, another app through MCP, a trigger's stored words).
    When the request names a service with a place for its secret in Settings (mail, Slack, Discord, GitHub), it says
    where; for anything else, that Heliox does not take secrets inside a request. ``words`` is the request with the
    secret already left out; nothing of the secret is in the sentence."""
    from pilot.system.connections import TITLE, WHERE

    service = next((name for name, pattern in _SERVICE_WORDS if pattern.search(words or "")), "")
    found = _LEFT_OUT_KIND.search(words or "")
    kind = found.group("kind") if found else "secret"
    kind = {"code": "code", "something private": "secret"}.get(kind, kind)
    if sent:
        where = f"Connect {TITLE[service]} once in {WHERE[service]}" if service else "Send it again without it"
        return f"I left out the {kind} in that request and did nothing: Heliox never puts a secret into a request. {where}."
    if not typed:
        if service:
            place = WHERE[service].replace(" > ", ", ")
            return (
                f"I don't use a PIN, password or code said aloud, so I left it out. Connect {TITLE[service]} once "
                f"in {place}, then ask again without it."
            )
        return SECRET_NOT_USED
    if service:
        return (
            f"I left out the {kind} you typed and did nothing: Heliox never puts a secret into a request. Connect "
            f"{TITLE[service]} once in {WHERE[service]}, then ask again without it."
        )
    return (
        f"I left out the {kind} you typed and did nothing: Heliox doesn't take passwords, keys or tokens inside a "
        "request, because the request is read by the language model that plans it. Ask again without it, or do this "
        "step yourself."
    )


__all__ = [
    "MARK_START",
    "SECRET_NOT_USED",
    "Withheld",
    "forget_recent",
    "holds_left_out",
    "holds_secret",
    "left_out",
    "secret_not_used",
    "withhold",
    "without_values",
]
