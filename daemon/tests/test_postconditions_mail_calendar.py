"""Mail and calendar post-conditions: read back over a new IMAP session or a fresh CalDAV client, all faked here.

No test opens a network connection: ``imaplib.IMAP4_SSL``, ``smtplib.SMTP`` and ``caldav.DAVClient`` are replaced by
small fakes, and the pauses between Sent-folder searches are recorded instead of waited.
"""

from __future__ import annotations

import imaplib
from email.parser import Parser
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    EmailParams,
)
from pilot.agents.email_agent import EmailAgent
from pilot.agents.postconditions import mail_calendar
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.config import PilotConfig
from pilot.system import mail_readback

GMAIL_SENT = "[Gmail]/Sent Mail"


class FakeMailServer:
    """One IMAP mailbox. ``files`` holds the raw headers of every message in each folder."""

    def __init__(self, *, folders: list[bytes] | None = None, appear_after_searches: int = 0) -> None:
        self.folders = folders or [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasChildren \\Noselect) "/" "[Gmail]"',
            b'(\\HasNoChildren \\Sent) "/" "[Gmail]/Sent Mail"',
        ]
        self.files: dict[str, list[str]] = {}
        self.fetch_status = "OK"
        self.appear_after_searches = appear_after_searches
        self.searches = 0
        self.sessions = 0
        self.refuse_login = False
        self.connect_error: Exception | None = None
        self.search_status = "OK"
        self.selected_read_only: list[bool] = []

    def file(self, folder: str, raw_headers: str) -> None:
        self.files.setdefault(folder, []).append(raw_headers)

    def imap_class(self) -> type:
        server = self

        class FakeIMAP:
            def __init__(self, host: str, ssl_context=None, timeout=None) -> None:
                if server.connect_error is not None:
                    raise server.connect_error
                server.sessions += 1
                self.selected = ""

            def login(self, user: str, password: str) -> tuple[str, list[bytes]]:
                if server.refuse_login:
                    raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")
                return "OK", [b"logged in"]

            def list(self) -> tuple[str, list[bytes]]:
                return "OK", list(server.folders)

            def select(self, mailbox: str, readonly: bool = False) -> tuple[str, list[bytes]]:
                server.selected_read_only.append(readonly)
                self.selected = mailbox.strip('"')
                return "OK", [b"1"]

            def uid(self, command: str, *args: object) -> tuple[str, list]:
                messages = server.files.get(self.selected, [])
                if command == "SEARCH":
                    server.searches += 1
                    if server.search_status != "OK":
                        return server.search_status, [b"SEARCH not allowed"]
                    if server.searches <= server.appear_after_searches:
                        return "OK", [b""]
                    wanted = str(args[-1]).strip('"')
                    hits = [
                        str(number + 1).encode()
                        for number, raw in enumerate(messages)
                        if wanted in (Parser().parsestr(raw, headersonly=True).get("Message-ID") or "")
                    ]
                    return "OK", [b" ".join(hits)]
                if command == "FETCH":
                    raw = messages[int(args[0]) - 1]
                    return "OK", [(b"1 (UID 1 BODY[HEADER.FIELDS] {10}", raw.encode()), b")"]
                raise AssertionError(f"the read-back must not run {command}")

            def fetch(self, number: str, parts: str) -> tuple[str, list]:
                assert "BODY.PEEK" in parts, "reading the original must not mark it as read"
                messages = server.files.get(self.selected, [])
                if server.fetch_status != "OK" or int(number) > len(messages):
                    return "NO", [None]
                return "OK", [(b"1 (BODY[HEADER.FIELDS] {10}", messages[int(number) - 1].encode()), b")"]

            def logout(self) -> tuple[str, list[bytes]]:
                return "BYE", [b""]

        return FakeIMAP


@pytest.fixture
def mail_server(monkeypatch: pytest.MonkeyPatch) -> FakeMailServer:
    server = FakeMailServer()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", server.imap_class())
    return server


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(mail_readback, "_pause", record)
    return waited


@pytest.fixture
def mail_settings(monkeypatch: pytest.MonkeyPatch) -> PilotConfig:
    config = PilotConfig()
    config.email.enabled = True
    config.email.imap_host = "imap.gmail.com"
    config.email.smtp_host = "smtp.gmail.com"
    config.email.username = "person@example.test"
    vault = MagicMock()
    vault.get_key = AsyncMock(return_value="app-password")
    monkeypatch.setattr(mail_calendar, "_config", config)
    monkeypatch.setattr(mail_calendar, "_vault", vault)
    return config


async def _verify(result: ActionResult):
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[result.action]), [result])


async def _detail(result: ActionResult) -> str:
    """The verifier's own words: the verification result shows them only for a MISMATCH."""
    from pilot.agents import postconditions

    return (await postconditions.registrations()[result.action.action_type].check(result)).detail


def _headers_of(sent: dict) -> str:
    """What a provider files in Sent: the headers of the message its SMTP server accepted."""
    return sent["msg"].split("\n\n", 1)[0]


async def _send_through_agent(
    smtp_mock: list, action_type: ActionType = ActionType.API_SEND_EMAIL, **fields: str
) -> ActionResult:
    agent = EmailAgent(AsyncMock())
    values = {
        "smtp_host": "smtp.gmail.com",
        "imap_host": "imap.gmail.com",
        "username": "person@example.test",
        "app_password": "app-password",
        "to": "friend@example.test",
        "subject": "Lunch",
        "reply_body": "Noon works.",
        **fields,
    }
    action = Action(action_type=action_type, parameters=EmailParams(**values))
    [result] = await agent.handle_task("send it", ActionPlan(actions=[action]))
    return result


# ── api_send_email ──────────────────────────────────────────────────────────


def test_mail_verifiers_are_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.API_SEND_EMAIL] == "conditional_sent_copy_postcondition"


@pytest.mark.asyncio
async def test_sent_message_found_in_sent_folder_verifies(mail_server, pauses, mail_settings, smtp_mock) -> None:
    result = await _send_through_agent(smtp_mock)
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert verification.details[0].startswith("Action 0 (api_send_email): VERIFIED")
    assert mail_server.selected_read_only == [True]
    assert pauses == []
    assert len(smtp_mock) == 1


@pytest.mark.asyncio
async def test_message_id_names_the_sender_domain_and_reaches_the_output(smtp_mock) -> None:
    result = await _send_through_agent(smtp_mock)

    sent = Parser().parsestr(smtp_mock[0]["msg"], headersonly=True)
    assert sent["Message-ID"].endswith("@example.test>")
    assert sent["Date"]
    assert mail_readback.message_id_from_output(result.output) == sent["Message-ID"]


@pytest.mark.asyncio
async def test_copy_missing_from_gmail_sent_is_a_mismatch_and_nothing_is_resent(
    mail_server, pauses, mail_settings, smtp_mock
) -> None:
    result = await _send_through_agent(smtp_mock)

    verification = await _verify(result)

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    detail = verification.details[0]
    assert "MISMATCH" in detail
    assert "after 4 searches over 14 s" in detail
    assert "Heliox did not send it again" in detail
    assert "Noon works." not in detail
    assert pauses == [2.0, 4.0, 8.0]
    assert len(smtp_mock) == 1


@pytest.mark.asyncio
async def test_copy_filed_late_is_found_by_a_later_search(mail_server, pauses, mail_settings, smtp_mock) -> None:
    result = await _send_through_agent(smtp_mock)
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))
    mail_server.appear_after_searches = 2

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert pauses == [2.0, 4.0]
    assert mail_server.searches == 3


@pytest.mark.asyncio
async def test_copy_missing_at_a_provider_that_may_not_file_smtp_mail_is_not_verified(
    mail_server, pauses, mail_settings, smtp_mock
) -> None:
    mail_settings.email.imap_host = "imap.example.test"
    mail_server.folders = [b'(\\HasNoChildren) "." "INBOX"', b'(\\HasNoChildren) "." "Sent Items"']
    result = await _send_through_agent(smtp_mock)

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_sent_folder_found_by_name_when_none_is_marked(mail_server, pauses, mail_settings, smtp_mock) -> None:
    mail_server.folders = [b'(\\HasNoChildren) "." INBOX', b'(\\HasNoChildren) "." "Sent Items"']
    result = await _send_through_agent(smtp_mock)
    mail_server.file("Sent Items", _headers_of(smtp_mock[0]))

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_configured_sent_folder_wins_and_a_missing_one_is_not_guessed(
    mail_server, pauses, mail_settings, smtp_mock
) -> None:
    mail_server.folders = [b'(\\HasNoChildren) "." INBOX', b'(\\HasNoChildren) "." "INBOX.Gesendet"']
    result = await _send_through_agent(smtp_mock)
    mail_server.file("INBOX.Gesendet", _headers_of(smtp_mock[0]))

    no_folder = await _verify(result)
    why = await _detail(result)
    mail_settings.email.sent_mailbox = "INBOX.Gesendet"
    configured = await _verify(result)

    assert no_folder.passed is True and no_folder.independently_observed_actions == []
    assert "email.sent_mailbox" in why
    assert configured.passed is True and configured.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_no_imap_settings_leave_the_send_a_receipt(mail_server, pauses, mail_settings, smtp_mock) -> None:
    mail_settings.email.enabled = False
    result = await _send_through_agent(smtp_mock)

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]
    assert mail_server.sessions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connect", "login", "search"])
async def test_a_search_that_cannot_be_made_is_not_verified_never_a_mismatch(
    failure, mail_server, pauses, mail_settings, smtp_mock
) -> None:
    result = await _send_through_agent(smtp_mock)
    if failure == "connect":
        mail_server.connect_error = OSError("network unreachable")
    elif failure == "login":
        mail_server.refuse_login = True
    else:
        mail_server.search_status = "NO"

    verification = await _verify(result)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "Not verified" in (await _detail(result))
    assert "app-password" not in (await _detail(result))
    assert pauses == []


@pytest.mark.asyncio
async def test_a_send_from_another_account_is_not_searched_for(mail_server, pauses, mail_settings, smtp_mock) -> None:
    mail_settings.email.username = "someone-else@example.test"
    result = await _send_through_agent(smtp_mock)

    verification = await _verify(result)

    assert verification.independently_observed_actions == []
    assert mail_server.sessions == 0


@pytest.mark.asyncio
async def test_send_without_credentials_is_a_failed_action(smtp_mock) -> None:
    agent = EmailAgent(AsyncMock())
    action = Action(
        action_type=ActionType.API_SEND_EMAIL,
        parameters=EmailParams(smtp_host="smtp.gmail.com", to="friend@example.test", reply_body="hi"),
    )
    [result] = await agent.handle_task("send it", ActionPlan(actions=[action]))

    verification = await _verify(result)

    assert result.success is False
    assert "FAILED" in verification.details[0]
    assert smtp_mock == []


@pytest.mark.asyncio
async def test_executor_send_names_its_message_id_and_is_read_back(
    mail_server, pauses, mail_settings, smtp_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pilot.system.api_client import send_email

    monkeypatch.setenv("SMTP_USER", "person@example.test")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    output = await send_email("friend@example.test", "Lunch", "Noon works.")
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))
    action = Action(
        action_type=ActionType.API_SEND_EMAIL,
        parameters=ApiRequestParams(to="friend@example.test", subject="Lunch", message="Noon works."),
    )

    verification = await _verify(ActionResult(action=action, success=True, output=output))

    sent = Parser().parsestr(smtp_mock[0]["msg"], headersonly=True)
    assert mail_readback.message_id_from_output(output) == sent["Message-ID"]
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_output_without_a_message_id_is_not_verified(mail_server, pauses, mail_settings) -> None:
    action = Action(action_type=ActionType.API_SEND_EMAIL, parameters=ApiRequestParams(to="friend@example.test"))

    verification = await _verify(ActionResult(action=action, success=True, output="Email sent to friend@example.test"))

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert mail_server.sessions == 0


def test_list_lines_with_literals_and_quotes_are_read() -> None:
    assert mail_readback._list_entry((b'(\\HasNoChildren \\Sent) "/" {9}', b"Gesendet")) == (
        {"\\hasnochildren", "\\sent"},
        "Gesendet",
    )
    assert mail_readback._list_entry(b'(\\HasNoChildren) "/" "My \\"Sent\\""') == ({"\\hasnochildren"}, 'My "Sent"')
    assert mail_readback._list_entry(b'(\\Noselect) "/" "[Gmail]"') is None
    assert mail_readback._list_entry(b"garbage") is None


# ── email_reply ─────────────────────────────────────────────────────────────

ORIGINAL_ID = "<original-1@example.test>"
ORIGINAL_HEADERS = (
    "Message-ID: <original-1@example.test>\n"
    "References: <earlier-0@example.test>\n"
    "From: Friend <friend@example.test>\n"
    "Reply-To: Friend Desk <desk@example.test>\n"
    "Subject: Lunch on Friday?"
)


def test_reply_verifier_is_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.EMAIL_REPLY] == "conditional_threaded_reply_postcondition"


@pytest.mark.asyncio
async def test_reply_threads_under_the_original_and_is_read_back(mail_server, pauses, mail_settings, smtp_mock) -> None:
    result = await _send_through_agent(smtp_mock, ActionType.EMAIL_REPLY, in_reply_to=ORIGINAL_ID)
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))

    verification = await _verify(result)

    sent = Parser().parsestr(smtp_mock[0]["msg"], headersonly=True)
    assert sent["In-Reply-To"] == ORIGINAL_ID
    assert sent["References"] == ORIGINAL_ID
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert "threads under <original-1@example.test>" in (await _detail(result))


@pytest.mark.asyncio
async def test_reply_by_inbox_number_reads_the_original_without_marking_it_read(
    mail_server, pauses, mail_settings, smtp_mock
) -> None:
    mail_server.file("INBOX", ORIGINAL_HEADERS)
    result = await _send_through_agent(smtp_mock, ActionType.EMAIL_REPLY, reply_to_uid="1", to="", subject="")
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))

    verification = await _verify(result)

    sent = Parser().parsestr(smtp_mock[0]["msg"], headersonly=True)
    assert smtp_mock[0]["to"] == ["desk@example.test"]
    assert sent["Subject"] == "Re: Lunch on Friday?"
    assert sent["In-Reply-To"] == ORIGINAL_ID
    assert sent["References"] == f"<earlier-0@example.test> {ORIGINAL_ID}"
    assert mail_server.selected_read_only == [True, True]
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_reply_whose_copy_lost_its_threading_is_a_mismatch(mail_server, pauses, mail_settings, smtp_mock) -> None:
    result = await _send_through_agent(smtp_mock, ActionType.EMAIL_REPLY, in_reply_to=ORIGINAL_ID)
    stripped = "\n".join(
        line for line in _headers_of(smtp_mock[0]).splitlines() if not line.startswith(("In-Reply-To", "References"))
    )
    mail_server.file(GMAIL_SENT, stripped)

    verification = await _verify(result)

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "In-Reply-To is missing instead of <original-1@example.test>" in verification.details[0]


@pytest.mark.asyncio
async def test_reply_to_an_unreadable_original_sends_nothing(mail_server, pauses, mail_settings, smtp_mock) -> None:
    mail_server.fetch_status = "NO"
    result = await _send_through_agent(smtp_mock, ActionType.EMAIL_REPLY, reply_to_uid="7", to="")

    verification = await _verify(result)

    assert result.success is False
    assert "nothing was sent" in (result.error or "")
    assert "FAILED" in verification.details[0]
    assert smtp_mock == []


@pytest.mark.asyncio
async def test_reply_that_named_no_original_is_checked_in_sent_only(
    mail_server, pauses, mail_settings, smtp_mock
) -> None:
    result = await _send_through_agent(smtp_mock, ActionType.EMAIL_REPLY)
    mail_server.file(GMAIL_SENT, _headers_of(smtp_mock[0]))

    verification = await _verify(result)

    sent = Parser().parsestr(smtp_mock[0]["msg"], headersonly=True)
    assert sent["In-Reply-To"] is None
    assert verification.passed is True
    assert "nothing to thread it under" in (await _detail(result))


@pytest.mark.asyncio
async def test_fetched_mail_reports_its_message_id_for_a_later_reply(imap_mock) -> None:
    import json

    agent = EmailAgent(AsyncMock())
    params = EmailParams(imap_host="imap.fake.com", username="user", app_password="pwd")
    [result] = await agent.handle_task(
        "fetch", ActionPlan(actions=[Action(action_type=ActionType.EMAIL_FETCH, parameters=params)])
    )

    assert all("message_id" in message for message in json.loads(result.output))
