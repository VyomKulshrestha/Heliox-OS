"""Read handlers that changed state unasked, hid part of their answer or passed a failure off as data. Fakes only."""

from __future__ import annotations

import pytest

from pilot.actions import Action, ActionType, EmailParams

# ── email_fetch ──────────────────────────────────────────────────────


class _Mailbox:
    """An IMAP server that sets \\Seen the way a real one does: on a non-PEEK body fetch, or on STORE +FLAGS."""

    def __init__(self, messages: dict[bytes, bytes]) -> None:
        self.messages = messages
        self.seen: set[bytes] = set()
        self.specs: list[str] = []

    def client(self):
        mailbox = self

        class _IMAP:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def login(self, *_args: object) -> None:
                pass

            def select(self, *_args: object) -> None:
                pass

            def search(self, *_args: object):
                unseen = [uid for uid in mailbox.messages if uid not in mailbox.seen]
                return "OK", [b" ".join(unseen)]

            def fetch(self, uid: bytes, spec: str):
                mailbox.specs.append(spec)
                if "PEEK" not in spec.upper():
                    mailbox.seen.add(uid)
                return "OK", [(uid + b" (BODY[] {1}", mailbox.messages[uid])]

            def store(self, uid: bytes, command: str, flag: str) -> None:
                if command == "+FLAGS" and "Seen" in flag:
                    mailbox.seen.add(uid)

            def logout(self) -> None:
                pass

        return _IMAP


def _message(subject: str, body: bytes = b"hello") -> bytes:
    return b"From: a@example.com\r\nSubject: " + subject.encode() + b"\r\n\r\n" + body


async def _fetch(monkeypatch: pytest.MonkeyPatch, mailbox: _Mailbox, **params: object):
    from pilot.agents import email_agent

    monkeypatch.setattr(email_agent.imaplib, "IMAP4_SSL", mailbox.client())
    agent = object.__new__(email_agent.EmailAgent)
    fetch = EmailParams(imap_host="imap.invalid", username="u", app_password="p", **params)
    action = Action(action_type=ActionType.EMAIL_FETCH, target="mail", parameters=fetch)
    return await agent._fetch_emails(action, fetch)


@pytest.mark.asyncio
async def test_email_fetch_leaves_mail_unread_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox = _Mailbox({b"1": _message("one"), b"2": _message("two")})

    result = await _fetch(monkeypatch, mailbox)

    assert result.success is True
    assert mailbox.seen == set()
    assert mailbox.specs and all("BODY.PEEK[]" in spec for spec in mailbox.specs)


@pytest.mark.asyncio
async def test_email_fetch_marks_mail_read_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox = _Mailbox({b"1": _message("one"), b"2": _message("two")})

    await _fetch(monkeypatch, mailbox, mark_as_read=True)

    assert mailbox.seen == {b"1", b"2"}
