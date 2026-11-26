"""Find a message Heliox sent in the sender's own Sent folder, over a new IMAP session.

The mail post-condition verifiers use this to read back what the SMTP server accepted. It only reads: the folder is
opened read-only (EXAMINE) and headers are fetched with BODY.PEEK, so no flag changes. It never sends, appends, moves
or deletes anything, and nothing here can send a message a second time.

A send names its message by the Message-ID header the sender chose (``new_message_id``) and reports it in its output
(``message_id_note``); the verifier takes that identity from the output and looks for it in the Sent folder.

Which folder is Sent differs by provider and language ("Sent", "[Gmail]/Sent Mail", "Sent Items", localised names).
It is found in this order, and the result says which rule found it:

1. the folder named in the ``email.sent_mailbox`` setting, when set;
2. the folder the server marks with the ``\\Sent`` special-use flag (RFC 6154);
3. the first existing folder whose name is one of ``SENT_FOLDER_NAMES``.

When none applies the read-back reports that it could not look; it never picks a folder by guessing.
"""

from __future__ import annotations

import asyncio
import email.utils
import imaplib
import logging
import re
import ssl
from dataclasses import dataclass, field, replace
from email.message import Message
from email.parser import BytesHeaderParser

logger = logging.getLogger("pilot.system.mail_readback")

# One IMAP step (connect, sign in, list, search) that takes longer than this is given up on.
SOCKET_TIMEOUT_SECONDS = 10.0

# Pauses before each search. A server can take a few seconds to file a message it accepted, so the search is repeated
# after 2, 4 and 8 more seconds: at most four searches and 14 s of waiting, inside the verifier's own time limit.
SEARCH_PAUSES_SECONDS = (0.0, 2.0, 4.0, 8.0)

SENT_FOLDER_NAMES = (
    "Sent",
    "Sent Items",
    "Sent Messages",
    "Sent Mail",
    "[Gmail]/Sent Mail",
    "[Google Mail]/Sent Mail",
    "INBOX.Sent",
    "INBOX/Sent",
)

# IMAP hosts whose provider files every message sent through its SMTP server in the Sent folder by itself. For other
# providers a missing copy proves nothing: many keep one only when the mail program appends it. Gmail's filing is its
# documented behaviour; for the Microsoft hosts (Outlook.com, Microsoft 365) it is the expected behaviour, not yet
# confirmed against a live account.
FILES_SMTP_MAIL_IN_SENT = frozenset(
    {
        "imap.gmail.com",
        "imap.googlemail.com",
        "outlook.office365.com",
        "imap-mail.outlook.com",
        "imap.outlook.com",
    }
)

_MESSAGE_ID_NOTE = re.compile(r"Message-ID (<[^<>\s]+>)")
_IN_REPLY_TO_NOTE = re.compile(r"in reply to (<[^<>\s]+>)")
_LIST_LINE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?P<delimiter>NIL|"(?:[^"\\]|\\.)*")\s*(?P<name>.*)$')
_LITERAL_SIZE = re.compile(rb"\{\d+\}$")

# Replaced in tests by a recorder, so no test waits in real time.
_pause = asyncio.sleep


def new_message_id(sender: str) -> str:
    """A Message-ID for a message from ``sender``, on the sender's own domain (never this computer's host name)."""
    domain = sender.rpartition("@")[2].strip().strip(">") if "@" in sender else ""
    return email.utils.make_msgid(domain=domain or "localhost")


def message_id_note(message_id: str, in_reply_to: str = "") -> str:
    """The part of a send's output that names the message, read back by ``message_id_from_output``."""
    note = f"Message-ID {message_id}"
    if in_reply_to:
        note += f", in reply to {in_reply_to}"
    return f" ({note})"


def message_id_from_output(output: str | None) -> str:
    match = _MESSAGE_ID_NOTE.search(output or "")
    return match.group(1) if match else ""


def in_reply_to_from_output(output: str | None) -> str:
    match = _IN_REPLY_TO_NOTE.search(output or "")
    return match.group(1) if match else ""


@dataclass(frozen=True)
class ImapAccount:
    host: str
    username: str
    password: str = field(repr=False)
    sent_mailbox: str = ""


@dataclass(frozen=True)
class SentCopy:
    """What the read-back saw.

    ``state`` is ``found``, ``absent`` (the search worked and the message is not there) or ``unavailable`` (the search
    could not be made; ``reason`` says why in words a person can act on).
    """

    state: str
    folder: str = ""
    found_by: str = ""
    reason: str = ""
    in_reply_to: str = ""
    references: str = ""
    searches: int = 0
    waited_seconds: float = 0.0


async def find_sent_copy(account: ImapAccount, message_id: str) -> SentCopy:
    """Search the Sent folder for ``message_id``, a few times while the server files it. Read-only."""
    waited = 0.0
    copy = SentCopy("unavailable", reason="no search was made")
    for attempt, pause in enumerate(SEARCH_PAUSES_SECONDS, start=1):
        if pause:
            await _pause(pause)
            waited += pause
        copy = replace(
            await asyncio.to_thread(_search_once, account, message_id), searches=attempt, waited_seconds=waited
        )
        if copy.state != "absent":
            return copy
    return copy


def _search_once(account: ImapAccount, message_id: str) -> SentCopy:
    try:
        connection = imaplib.IMAP4_SSL(
            account.host,
            ssl_context=ssl.create_default_context(),
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except (OSError, imaplib.IMAP4.error) as exc:
        return SentCopy("unavailable", reason=f"could not connect to {account.host} ({type(exc).__name__})")
    try:
        try:
            connection.login(account.username, account.password)
        except imaplib.IMAP4.error:
            return SentCopy("unavailable", reason=f"{account.host} refused the saved IMAP sign-in")
        folder, found_by, why = _sent_folder(connection, account.sent_mailbox)
        if not folder:
            return SentCopy("unavailable", reason=why)
        status, _ = connection.select(_quoted(folder), readonly=True)
        if status != "OK":
            return SentCopy("unavailable", folder=folder, reason=f'could not open the folder "{folder}"')
        status, data = connection.uid("SEARCH", "HEADER", "Message-ID", _quoted(message_id))
        if status != "OK":
            return SentCopy("unavailable", folder=folder, reason=f'{account.host} could not search "{folder}"')
        uids = data[0].split() if data and isinstance(data[0], bytes) else []
        for uid in uids:
            headers = _fetch_headers(connection, uid)
            if headers is not None and str(headers.get("Message-ID", "")).strip() == message_id:
                return SentCopy(
                    "found",
                    folder=folder,
                    found_by=found_by,
                    in_reply_to=str(headers.get("In-Reply-To", "") or "").strip(),
                    references=" ".join(str(headers.get("References", "") or "").split()),
                )
        return SentCopy("absent", folder=folder, found_by=found_by)
    except (OSError, imaplib.IMAP4.error, EOFError) as exc:
        logger.info("Sent-folder read-back on %s failed: %s", account.host, type(exc).__name__)
        return SentCopy("unavailable", reason=f"the IMAP session to {account.host} failed ({type(exc).__name__})")
    finally:
        try:
            connection.logout()
        except Exception:  # noqa: BLE001
            pass


def _sent_folder(connection: imaplib.IMAP4, configured: str) -> tuple[str, str, str]:
    """(folder, how it was found, why none was found)."""
    status, data = connection.list()
    if status != "OK":
        return "", "", "the server would not list its folders"
    folders = [entry for entry in (_list_entry(item) for item in data or []) if entry is not None]
    if configured:
        for _flags, name in folders:
            if name == configured:
                return name, "set in email.sent_mailbox", ""
        return "", "", f'the folder set in email.sent_mailbox ("{configured}") is not on the server'
    for flags, name in folders:
        if "\\sent" in flags:
            return name, "marked \\Sent by the server", ""
    by_name = {name.casefold(): name for _flags, name in folders}
    for candidate in SENT_FOLDER_NAMES:
        if candidate.casefold() in by_name:
            return by_name[candidate.casefold()], "found by its name", ""
    return (
        "",
        "",
        "no folder is marked \\Sent and none has a usual Sent name; set email.sent_mailbox in config.toml",
    )


def _list_entry(item: object) -> tuple[set[str], str] | None:
    """(lower-case flags, folder name) from one LIST response line, or None for a line that is not one."""
    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], bytes) and isinstance(item[1], bytes):
        head = _LITERAL_SIZE.sub(b"", item[0].rstrip()).decode("utf-8", "replace")
        literal_name: str | None = item[1].decode("utf-8", "replace")
    elif isinstance(item, bytes):
        head = item.decode("utf-8", "replace")
        literal_name = None
    else:
        return None
    match = _LIST_LINE.match(head.strip())
    if match is None:
        return None
    flags = {flag.casefold() for flag in match.group("flags").split()}
    if "\\noselect" in flags or "\\nonexistent" in flags:
        return None
    name = literal_name if literal_name is not None else match.group("name").strip()
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = re.sub(r"\\(.)", r"\1", name[1:-1])
    return (flags, name) if name else None


def _fetch_headers(connection: imaplib.IMAP4, uid: bytes) -> Message | None:
    status, data = connection.uid("FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID IN-REPLY-TO REFERENCES)])")
    if status != "OK":
        return None
    for part in data or []:
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
            return BytesHeaderParser().parsebytes(part[1])
    return None


def _quoted(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
