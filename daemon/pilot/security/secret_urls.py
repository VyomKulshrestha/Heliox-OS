"""One way to hide the secret in a URL wherever a plan or its result is shown or kept.

A chat webhook URL (Slack, Discord, any ``webhook_url``) is itself the credential: whoever has it can post. A URL
can also carry a user and password, or a token in its query. Such a URL is shown and stored masked: the scheme, the
host and the last four characters, for example ``https://hooks.slack.com/***x9Qa``. The full value exists only in
the memory of the running plan, for the step that uses it. A masked URL can never be sent: the HTTP layer refuses
it (``refuse_masked``), so a plan restored from storage asks for the address again instead of posting to garbage.

Pure functions, no imports from the rest of Heliox, so ``pilot.actions`` can use them in its serializers.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

MASK = "***"

# Query keys whose value is a credential (compared case-insensitively, also as a suffix such as "x-amz-signature").
_SECRET_QUERY_KEYS = (
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "key",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "auth",
    "sig",
    "signature",
    "credential",
    "code",
    "session",
    "sessionid",
)
# Paths that are a webhook's address and therefore its secret.
_WEBHOOK_PATHS = (
    re.compile(r"^/(services|triggers|workflows)/", re.IGNORECASE),  # hooks.slack.com
    re.compile(r"/api(/v\d+)?/webhooks/\d+/", re.IGNORECASE),  # Discord
    re.compile(r"/(hooks?|webhooks?)/.*[A-Za-z0-9_-]{16,}", re.IGNORECASE),  # Zapier, Teams, Make, n8n and similar
)
_URL_IN_TEXT = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>`]+", re.IGNORECASE)


def is_masked(url: str | None) -> bool:
    return bool(url) and MASK in str(url)


def _secret_query_key(key: str) -> bool:
    lowered = key.casefold()
    return any(lowered == name or lowered.endswith(("-" + name, "_" + name)) for name in _SECRET_QUERY_KEYS)


def carries_secret(url: str | None) -> bool:
    """Whether the URL holds a credential: a user or password, a token in the query, or a webhook path."""
    if not url or is_masked(url):
        return False
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None:
        return True
    if any(_secret_query_key(key) for key, _ in parse_qsl(parts.query, keep_blank_values=True)):
        return True
    return any(pattern.search(parts.path or "") for pattern in _WEBHOOK_PATHS)


def mask_url(url: str | None) -> str | None:
    """``scheme://host/***last4``; the host is kept so a person can tell which service it was."""
    if not url or is_masked(url):
        return url
    text = str(url).strip()
    tail = text[-4:] if len(text) > 8 else ""
    try:
        parts = urlsplit(text)
        scheme, host = parts.scheme, parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
    except ValueError:
        scheme = host = ""
    if scheme and host:
        return f"{scheme}://{host}/{MASK}{tail}"
    return f"{MASK}{tail}"


def mask_if_secret(url: str | None) -> str | None:
    return mask_url(url) if carries_secret(url) else url


def redact_text(text: str | None) -> str | None:
    """The text with every URL that carries a secret masked; other text unchanged."""
    if not text or "://" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        trimmed = found.rstrip(".,;:)]}")
        rest = found[len(trimmed) :]
        return (mask_url(trimmed) or "") + rest if carries_secret(trimmed) else found

    return _URL_IN_TEXT.sub(replace, text)


# Keys whose value is always a secret address, whatever it looks like.
ALWAYS_SECRET_KEYS = frozenset({"webhook_url"})


def redact(value: Any) -> Any:
    """A copy of a JSON-like value with secret URLs masked: under ``webhook_url`` always, elsewhere when they carry
    a credential."""
    if isinstance(value, dict):
        return {
            key: (mask_url(item) if key in ALWAYS_SECRET_KEYS and isinstance(item, str) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def refuse_masked(url: str) -> None:
    """Raise when asked to send to an address that was masked: its secret is no longer known."""
    if is_masked(url):
        raise ValueError(
            f"The address {url} was masked when the plan was kept, so its secret is no longer known and nothing "
            "was sent. Give the address again, or set it in the daemon's settings."
        )


class SecretUrlFilter(logging.Filter):
    """Masks secret URLs in a log record's message and traceback before any handler writes it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string is logging's own problem; leave the record as it is
            return True
        record.msg = redact_text(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True
